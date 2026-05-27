import argparse
import collections
import collections.abc
import csv
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "NATIVE_AD_COMPARE "


collections.Callable = collections.abc.Callable


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _rss_mb() -> float | None:
    if os.name != "nt":
        return None
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    try:
        psapi = ctypes.WinDLL("psapi.dll")
        kernel32 = ctypes.WinDLL("kernel32.dll")
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return None


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


def _stats_ms(samples):
    return {
        "samples": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _values_f32(n: int):
    return (np.arange(n, dtype=np.float32) % 23).astype(np.float32)


def _indices(n: int):
    return (n - 1 - np.arange(n, dtype=np.int32)).astype(np.int32)


def _keys(n: int, groups: int):
    return (np.arange(n, dtype=np.int32) % groups).astype(np.int32)


def _arch_value(ti, name: str):
    if name == "cpu":
        return ti.cpu
    if name == "cuda":
        return ti.cuda
    if name == "vulkan":
        return ti.vulkan
    raise ValueError(name)


def _sync(ti):
    try:
        ti.sync()
    except Exception:
        pass


def _copy_method(ti):
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cuda:
        return "cuda_device"
    if arch == ti.vulkan:
        return "vulkan_native"
    return "cpu_native"


def _reduce_method(ti):
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cuda:
        return "cuda_cub"
    if arch == ti.vulkan:
        return "vulkan_native"
    return "cpu_native"


def _workspace_snapshot(ti):
    rows = {}
    prog = ti.lang.impl.get_runtime().prog
    for name in (
        "cuda_cub_scan_workspace_bytes",
        "cuda_cub_reduce_workspace_bytes",
        "vulkan_scan_workspace_bytes",
        "vulkan_reduce_workspace_bytes",
        "cpu_scan_workspace_bytes",
    ):
        if hasattr(prog, name):
            try:
                rows[name] = int(getattr(prog, name)())
            except Exception as exc:  # pragma: no cover
                rows[name] = repr(exc)
    return rows


def _clear_grad(obj):
    grad = getattr(obj, "grad", None)
    if grad is not None:
        grad.fill(0)


def _to_numpy(obj):
    return obj.to_numpy()


def _first_value(obj):
    if hasattr(obj, "to_numpy"):
        return float(obj.to_numpy().reshape(-1)[0])
    return float(obj[0])


def _make_bodies(ti, package: str, container: str, n: int, mode: str, op: str):
    ad = mode == "ad"
    groups = min(256, max(1, n))
    idx_np = _indices(n)
    values_np = _values_f32(n)
    ones_np = np.ones(n, dtype=np.float32)
    keys_np = _keys(n, groups)

    if container == "field":
        values = ti.field(ti.f32, shape=n, needs_grad=ad)
        src = ti.field(ti.f32, shape=n, needs_grad=ad)
        dst = ti.field(ti.f32, shape=n, needs_grad=ad)
        indices = ti.field(ti.i32, shape=n)
        keys = ti.field(ti.i32, shape=n)
        grouped = ti.field(ti.f32, shape=groups, needs_grad=ad)
        output = ti.field(ti.f32, shape=(), needs_grad=ad)
        loss = ti.field(ti.f32, shape=(), needs_grad=ad)
    elif container == "ndarray":
        values = ti.ndarray(ti.f32, shape=n, needs_grad=ad)
        src = ti.ndarray(ti.f32, shape=n, needs_grad=ad)
        dst = ti.ndarray(ti.f32, shape=n, needs_grad=ad)
        indices = ti.ndarray(ti.i32, shape=n)
        keys = ti.ndarray(ti.i32, shape=n)
        grouped = ti.ndarray(ti.f32, shape=groups, needs_grad=ad)
        output = ti.ndarray(ti.f32, shape=1, needs_grad=ad)
        loss = ti.field(ti.f32, shape=(), needs_grad=ad)
    else:
        raise ValueError(container)

    @ti.kernel
    def vanilla_transform_field():
        for i in range(n):
            dst[i] = src[i] * 2.5 + 1.0

    @ti.kernel
    def vanilla_transform_ndarray(
        src_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dst_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(n):
            dst_arr[i] = src_arr[i] * 2.5 + 1.0

    @ti.kernel
    def vanilla_gather_field():
        for i in range(n):
            dst[i] = src[indices[i]]

    @ti.kernel
    def vanilla_gather_ndarray(
        src_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        idx_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dst_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(n):
            dst_arr[i] = src_arr[idx_arr[i]]

    @ti.kernel
    def vanilla_scatter_field():
        for i in range(n):
            dst[indices[i]] = src[i]

    @ti.kernel
    def vanilla_scatter_ndarray(
        src_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        idx_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dst_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(n):
            dst_arr[idx_arr[i]] = src_arr[i]

    @ti.kernel
    def vanilla_scatter_add_field():
        for i in range(n):
            ti.atomic_add(dst[indices[i]], src[i])

    @ti.kernel
    def vanilla_scatter_add_ndarray(
        src_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        idx_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dst_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(n):
            ti.atomic_add(dst_arr[idx_arr[i]], src_arr[i])

    @ti.kernel
    def vanilla_reduce_field():
        for i in range(n):
            ti.atomic_add(output[None], values[i])

    @ti.kernel
    def vanilla_reduce_ndarray(
        arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(n):
            ti.atomic_add(out[0], arr[i])

    @ti.kernel
    def vanilla_grouped_reduce_field():
        for i in range(n):
            ti.atomic_add(grouped[keys[i]], values[i])

    @ti.kernel
    def vanilla_grouped_reduce_ndarray(
        key_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        val_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
        out_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(n):
            ti.atomic_add(out_arr[key_arr[i]], val_arr[i])

    @ti.kernel
    def sum_dst_field():
        for i in range(n):
            loss[None] += dst[i]

    @ti.kernel
    def sum_dst_ndarray(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            loss[None] += arr[i]

    @ti.kernel
    def sum_values_field():
        for i in range(n):
            loss[None] += values[i]

    @ti.kernel
    def sum_values_ndarray(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            loss[None] += arr[i]

    @ti.kernel
    def sum_output_field():
        loss[None] = output[None]

    @ti.kernel
    def sum_output_ndarray(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        loss[None] = out[0]

    @ti.kernel
    def sum_grouped_field():
        for i in range(groups):
            loss[None] += grouped[i]

    @ti.kernel
    def sum_grouped_ndarray(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(groups):
            loss[None] += out[i]

    def setup():
        values.from_numpy(values_np)
        src.from_numpy(values_np)
        dst.fill(0)
        indices.from_numpy(idx_np)
        keys.from_numpy(keys_np)
        grouped.fill(0)
        output.fill(0)
        loss.fill(0)
        if ad:
            _clear_grad(values)
            _clear_grad(src)
            _clear_grad(dst)
            _clear_grad(grouped)
            _clear_grad(output)
            _clear_grad(loss)

    scan_executor = ti.algorithms.PrefixSumExecutor(n)
    workspace = None
    if package == "forge" and op in ("gather", "scatter"):
        workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)
    elif package == "forge" and op == "scatter_add":
        workspace = ti.algorithms.ScatterAddWorkspace(max_items=n, max_groups=n)
    elif package == "forge" and op == "reduce":
        workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    elif package == "forge" and op == "grouped_reduce":
        workspace = ti.algorithms.GroupedReduceWorkspace(max_items=n, max_groups=groups)
    elif package == "forge" and op == "transform":
        workspace = ti.algorithms.TransformWorkspace(max_items=n)

    def plain_body():
        if package == "forge":
            if op == "transform":
                ti.algorithms.experimental_transform(
                    src, dst, scale=2.5, bias=1.0, method=_copy_method(ti),
                    workspace=workspace,
                )
            elif op == "gather":
                ti.algorithms.experimental_gather(
                    src, indices, dst, method=_copy_method(ti), workspace=workspace
                )
            elif op == "scatter":
                ti.algorithms.experimental_scatter(
                    src, indices, dst, method=_copy_method(ti), workspace=workspace
                )
            elif op == "scatter_add":
                ti.algorithms.experimental_scatter_add(
                    src, indices, dst, method=_copy_method(ti), workspace=workspace
                )
            elif op == "reduce":
                ti.algorithms.experimental_reduce(
                    values, output, op="sum", method=_reduce_method(ti),
                    workspace=workspace,
                )
            elif op == "grouped_reduce":
                ti.algorithms.experimental_grouped_reduce(
                    keys, values, grouped, op="sum", method=_copy_method(ti),
                    workspace=workspace,
                )
            elif op == "scan":
                scan_executor.run(values)
            else:
                raise ValueError(op)
            return
        if container == "field":
            if op == "transform":
                vanilla_transform_field()
            elif op == "gather":
                vanilla_gather_field()
            elif op == "scatter":
                vanilla_scatter_field()
            elif op == "scatter_add":
                vanilla_scatter_add_field()
            elif op == "reduce":
                vanilla_reduce_field()
            elif op == "grouped_reduce":
                vanilla_grouped_reduce_field()
            elif op == "scan":
                scan_executor.run(values)
            else:
                raise ValueError(op)
            return
        if op == "transform":
            vanilla_transform_ndarray(src, dst)
        elif op == "gather":
            vanilla_gather_ndarray(src, indices, dst)
        elif op == "scatter":
            vanilla_scatter_ndarray(src, indices, dst)
        elif op == "scatter_add":
            vanilla_scatter_add_ndarray(src, indices, dst)
        elif op == "reduce":
            vanilla_reduce_ndarray(values, output)
        elif op == "grouped_reduce":
            vanilla_grouped_reduce_ndarray(keys, values, grouped)
        elif op == "scan":
            scan_executor.run(values)
        else:
            raise ValueError(op)

    def ad_body():
        with ti.ad.Tape(loss):
            plain_body()
            if op in ("transform", "gather", "scatter", "scatter_add"):
                if container == "field":
                    sum_dst_field()
                else:
                    sum_dst_ndarray(dst)
            elif op in ("scan",):
                if container == "field":
                    sum_values_field()
                else:
                    sum_values_ndarray(values)
            elif op == "reduce":
                if container == "field":
                    sum_output_field()
                else:
                    sum_output_ndarray(output)
            elif op == "grouped_reduce":
                if container == "field":
                    sum_grouped_field()
                else:
                    sum_grouped_ndarray(grouped)
            else:
                raise ValueError(op)

    def body():
        if ad:
            ad_body()
        else:
            plain_body()

    def check_plain():
        if op == "transform":
            expected = values_np * 2.5 + 1.0
            actual = _to_numpy(dst).reshape(-1)
        elif op == "gather":
            expected = values_np[idx_np]
            actual = _to_numpy(dst).reshape(-1)
        elif op == "scatter":
            expected = np.zeros(n, dtype=np.float32)
            expected[idx_np] = values_np
            actual = _to_numpy(dst).reshape(-1)
        elif op == "scatter_add":
            expected = np.zeros(n, dtype=np.float32)
            np.add.at(expected, idx_np, values_np)
            actual = _to_numpy(dst).reshape(-1)
        elif op == "reduce":
            expected = np.array([values_np.sum()], dtype=np.float32)
            actual = _to_numpy(output).reshape(-1)[:1]
        elif op == "grouped_reduce":
            expected = np.zeros(groups, dtype=np.float32)
            np.add.at(expected, keys_np, values_np)
            actual = _to_numpy(grouped).reshape(-1)
        elif op == "scan":
            expected = np.cumsum(values_np, dtype=np.float32)
            actual = _to_numpy(values).reshape(-1)
        else:
            raise ValueError(op)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def check_ad():
        check_plain()
        if op == "transform":
            expected = np.full(n, 2.5, dtype=np.float32)
            actual = _to_numpy(src.grad).reshape(-1)
        elif op in ("gather", "scatter", "scatter_add"):
            expected = ones_np
            actual = _to_numpy(src.grad).reshape(-1)
        elif op == "reduce":
            expected = ones_np
            actual = _to_numpy(values.grad).reshape(-1)
        elif op == "grouped_reduce":
            expected = ones_np
            actual = _to_numpy(values.grad).reshape(-1)
        elif op == "scan":
            expected = np.arange(n, 0, -1, dtype=np.float32)
            actual = _to_numpy(values.grad).reshape(-1)
        else:
            raise ValueError(op)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def check():
        if ad:
            check_ad()
        else:
            check_plain()

    def workspace_bytes():
        peak = 0
        if workspace is not None:
            peak = max(peak, int(getattr(workspace, "workspace_bytes_peak", 0)))
        if op == "scan" and hasattr(scan_executor, "workspace_length"):
            peak = max(peak, int(scan_executor.workspace_length) * 4)
        if package == "forge":
            snap = _workspace_snapshot(ti)
            peak = max(peak, *(int(v) for v in snap.values() if isinstance(v, int)))
        return peak

    return setup, body, check, workspace_bytes


def _run_timed(ti, package, container, n, mode, op, repeats, warmups):
    setup, body, check, workspace_bytes = _make_bodies(
        ti, package, container, n, mode, op
    )
    rss_before = _rss_mb()
    gpu_before = _powershell_gpu_process_dedicated_mb(os.getpid())
    setup()
    _sync(ti)
    start = time.perf_counter()
    body()
    _sync(ti)
    first_ms = (time.perf_counter() - start) * 1000.0
    check()
    gpu_after_first = _powershell_gpu_process_dedicated_mb(os.getpid())
    rss_after_first = _rss_mb()
    for _ in range(warmups):
        setup()
        _sync(ti)
        body()
        _sync(ti)
        check()
    samples = []
    rss_values = [x for x in (rss_before, rss_after_first) if x is not None]
    rss_peak = max(rss_values) if rss_values else None
    gpu_values = [x for x in (gpu_before, gpu_after_first) if x is not None]
    gpu_peak = max(gpu_values) if gpu_values else None
    for _ in range(repeats):
        setup()
        _sync(ti)
        start = time.perf_counter()
        body()
        _sync(ti)
        samples.append((time.perf_counter() - start) * 1000.0)
        check()
        rss_now = _rss_mb()
        if rss_now is not None:
            rss_peak = rss_now if rss_peak is None else max(rss_peak, rss_now)
    gpu_after_run = _powershell_gpu_process_dedicated_mb(os.getpid())
    if gpu_after_run is not None:
        gpu_peak = gpu_after_run if gpu_peak is None else max(gpu_peak, gpu_after_run)
    stats = _stats_ms(samples)
    return {
        "package": package,
        "package_version": ".".join(map(str, getattr(ti, "__version__", (0, 0, 0))[:3])),
        "arch": _arch_name(ti),
        "container": container,
        "mode": mode,
        "op": op,
        "n": n,
        "status": "ok",
        "first_call_ms": first_ms,
        "runtime": stats,
        "workspace_peak_bytes": int(workspace_bytes()),
        "process_rss_mb": {
            "before": rss_before,
            "after_first": rss_after_first,
            "peak": rss_peak,
            "peak_delta": None
            if rss_before is None or rss_peak is None
            else rss_peak - rss_before,
        },
        "gpu_dedicated_mb": {
            "before": gpu_before,
            "after_first": gpu_after_first,
            "after_run": gpu_after_run,
            "peak": gpu_peak,
            "peak_delta": None
            if gpu_before is None or gpu_peak is None
            else gpu_peak - gpu_before,
        },
    }


def _arch_name(ti):
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cpu:
        return "cpu"
    if arch == ti.cuda:
        return "cuda"
    if arch == ti.vulkan:
        return "vulkan"
    return str(arch)


def run_child(args):
    if args.package == "forge":
        sys.path.insert(0, str(ROOT / "python"))
        import taichi_forge as ti  # pylint: disable=import-outside-toplevel
    else:
        import taichi as ti  # pylint: disable=import-outside-toplevel

    ti.init(arch=_arch_value(ti, args.arch), offline_cache=False)
    rows = []
    for op in args.ops:
        try:
            rows.append(
                _run_timed(
                    ti,
                    args.package,
                    args.container,
                    args.n,
                    args.mode,
                    op,
                    args.repeats,
                    args.warmups,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            rows.append(
                {
                    "package": args.package,
                    "package_version": ".".join(
                        map(str, getattr(ti, "__version__", (0, 0, 0))[:3])
                    ),
                    "arch": args.arch,
                    "container": args.container,
                    "mode": args.mode,
                    "op": op,
                    "n": args.n,
                    "status": "error",
                    "error": repr(exc),
                }
            )
    print(PREFIX + json.dumps(rows, ensure_ascii=False))
    return 0


def _write_summary(output_dir: Path, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "package",
        "package_version",
        "arch",
        "container",
        "mode",
        "op",
        "n",
        "status",
        "first_call_ms",
        "runtime_mean_ms",
        "runtime_median_ms",
        "workspace_peak_bytes",
        "process_peak_delta_mb",
        "gpu_peak_delta_mb",
        "error",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            runtime = row.get("runtime", {})
            process = row.get("process_rss_mb", {})
            gpu = row.get("gpu_dedicated_mb", {})
            writer.writerow(
                {
                    "package": row.get("package"),
                    "package_version": row.get("package_version"),
                    "arch": row.get("arch"),
                    "container": row.get("container"),
                    "mode": row.get("mode"),
                    "op": row.get("op"),
                    "n": row.get("n"),
                    "status": row.get("status"),
                    "first_call_ms": row.get("first_call_ms"),
                    "runtime_mean_ms": runtime.get("mean_ms"),
                    "runtime_median_ms": runtime.get("median_ms"),
                    "workspace_peak_bytes": row.get("workspace_peak_bytes"),
                    "process_peak_delta_mb": process.get("peak_delta"),
                    "gpu_peak_delta_mb": gpu.get("peak_delta"),
                    "error": row.get("error"),
                }
            )


def run_matrix(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    script = Path(__file__).resolve()
    for package in args.packages:
        for mode in args.modes:
            for container in args.containers:
                for arch in args.arches:
                    for n in args.sizes:
                        cmd = [
                            sys.executable,
                            str(script),
                            "--child",
                            "--package",
                            package,
                            "--mode",
                            mode,
                            "--container",
                            container,
                            "--arch",
                            arch,
                            "--n",
                            str(n),
                            "--repeats",
                            str(args.repeats),
                            "--warmups",
                            str(args.warmups),
                            "--ops",
                            *args.ops,
                        ]
                        key = f"{package}_{mode}_{container}_{arch}_{n}"
                        proc = subprocess.run(
                            cmd,
                            cwd=str(ROOT),
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        (output_dir / f"{key}.stdout.txt").write_text(
                            proc.stdout, encoding="utf-8"
                        )
                        (output_dir / f"{key}.stderr.txt").write_text(
                            proc.stderr, encoding="utf-8"
                        )
                        parsed = False
                        for line in proc.stdout.splitlines():
                            if line.startswith(PREFIX):
                                rows.extend(json.loads(line[len(PREFIX) :]))
                                parsed = True
                        if not parsed:
                            rows.append(
                                {
                                    "package": package,
                                    "arch": arch,
                                    "container": container,
                                    "mode": mode,
                                    "n": n,
                                    "status": "child_error",
                                    "error": proc.stderr[-2000:],
                                }
                            )
                        _write_summary(output_dir, rows)
                        print(
                            f"{key}: return={proc.returncode}, "
                            f"rows={len(rows)}"
                        )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--package", choices=("forge", "vanilla"), default="forge")
    parser.add_argument("--mode", choices=("plain", "ad"), default="plain")
    parser.add_argument("--container", choices=("field", "ndarray"), default="field")
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--ops",
        nargs="+",
        default=[
            "transform",
            "gather",
            "scatter",
            "scatter_add",
            "reduce",
            "scan",
            "grouped_reduce",
        ],
    )
    parser.add_argument("--output", default="benchmarks/results/native_ad_compare")
    parser.add_argument("--packages", nargs="+", default=["forge", "vanilla"])
    parser.add_argument("--modes", nargs="+", default=["plain", "ad"])
    parser.add_argument("--containers", nargs="+", default=["field", "ndarray"])
    parser.add_argument("--arches", nargs="+", default=["cpu", "cuda", "vulkan"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[2048, 65536])
    args = parser.parse_args()
    if args.child:
        return run_child(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
