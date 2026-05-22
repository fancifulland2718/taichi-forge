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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "S4_NATIVE_PLAN_REPLAY "


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


def _values(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.int32) % 17 - 8).astype(np.int32)


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
        return "cuda_cub" if op_name in ("scan", "reduce") else "cuda_device"
    if arch_name == "vulkan":
        return "vulkan_native"
    raise ValueError(arch_name)


def _sync(ti) -> None:
    try:
        ti.sync()
    except Exception:
        pass


def _available(ti, arch_name: str, op_name: str, storage: str) -> tuple[bool, str]:
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if storage == "struct_tensor_member" and op_name != "transform":
        return False, "tensor-member replay plan is transform-only"
    if arch_name == "cpu":
        if storage == "struct_tensor_member":
            name = "cpu_transform_affine_packed_strided_ndarray"
            return (
                hasattr(prog, "cpu_transform_available")
                and prog.cpu_transform_available()
                and hasattr(prog, name)
            ), name
        names = {
            "scan": "cpu_scan_available",
            "reduce": "cpu_reduce_available",
            "transform": "cpu_transform_available",
        }
        name = names[op_name]
        return hasattr(prog, name) and getattr(prog, name)(), name
    if arch_name == "cuda":
        if op_name == "scan":
            name = "cuda_cub_scan_available"
        elif op_name == "reduce":
            name = "cuda_cub_reduce_available"
        elif storage in ("struct_member", "struct_tensor_member"):
            name = "cuda_toolkit_transform_available"
        else:
            name = "cuda_device_transform_available"
        ok = hasattr(prog, name) and getattr(prog, name)()
        if storage == "struct_tensor_member":
            packed = "cuda_device_transform_affine_packed_strided_ndarray"
            ok = ok and hasattr(prog, packed)
            name = packed
        return ok, name
    if arch_name == "vulkan":
        names = {
            "scan": "vulkan_scan_available",
            "reduce": "vulkan_reduce_available",
            "transform": "vulkan_transform_available",
        }
        name = names[op_name]
        if not (hasattr(prog, name) and getattr(prog, name)()):
            return False, name
        value_gate = {
            "scan": "vulkan_scan_value_type_available",
            "reduce": "vulkan_reduce_value_type_available",
            "transform": "vulkan_transform_value_type_available",
        }[op_name]
        if hasattr(prog, value_gate) and not getattr(prog, value_gate)(0):
            return False, value_gate
        if storage == "struct_tensor_member":
            packed = "vulkan_transform_affine_packed_strided_ndarray"
            if not hasattr(prog, packed):
                return False, packed
        return True, name
    return False, arch_name


def _runtime_workspace_peak(arch_name: str, op_name: str) -> int:
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    candidates = []
    if op_name == "scan":
        candidates = [
            f"{arch_name}_scan_workspace_bytes",
            "cuda_cub_scan_workspace_bytes",
        ]
    elif op_name == "reduce":
        candidates = [
            f"{arch_name}_reduce_workspace_bytes",
            "cuda_cub_reduce_workspace_bytes",
        ]
    for name in candidates:
        if hasattr(prog, name):
            return int(getattr(prog, name)())
    return 0


def _make_storage(ti, storage: str, n: int):
    data = _values(n)
    if storage == "field":
        src = ti.field(ti.i32, shape=n)
        src.from_numpy(data)
        return src, src, None, data
    if storage == "ndarray":
        src = ti.ndarray(ti.i32, shape=n)
        src.from_numpy(data)
        return src, src, None, data
    if storage == "struct_member":
        payload = ti.types.struct(value=ti.i32, tag=ti.i32)
        base = ti.ndarray(payload, shape=n)
        host = np.zeros((n,), dtype=base.numpy_dtype)
        host["value"] = data
        host["tag"] = np.arange(n, dtype=np.int32) * 5 + 7
        base.from_numpy(host)
        return base.field("value"), base, host, data
    if storage == "struct_tensor_member":
        payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
        base = ti.ndarray(payload, shape=n)
        data = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 17 - 8).astype(
            np.int32
        )
        host = np.zeros((n,), dtype=base.numpy_dtype)
        host["vec"] = data
        host["tag"] = np.arange(n, dtype=np.int32) * 5 + 7
        base.from_numpy(host)
        return base.field("vec"), base, host, data
    raise ValueError(storage)


def _make_body(ti, arch_name: str, op_name: str, storage: str, n: int):
    method = _method_for(arch_name, op_name)
    src, src_owner, host, data = _make_storage(ti, storage, n)
    workspace = None
    executor = None
    dst_owner = None

    if op_name == "scan":
        executor = ti.algorithms.PrefixSumExecutor(n)

        def body():
            executor.run(src)

        def plan():
            return executor._native_scan_plan

        def verify():
            if storage == "field":
                actual = src_owner.to_numpy()
            elif storage == "ndarray":
                actual = src_owner.to_numpy()
            else:
                result = src_owner.to_numpy()
                if not np.array_equal(result["tag"], host["tag"]):
                    return False
                actual = result["value"]
            expected = np.cumsum(data, dtype=np.int64).astype(np.int32)
            return bool(np.array_equal(actual, expected))

        return body, plan, verify, lambda: _runtime_workspace_peak(arch_name, op_name)

    if op_name == "reduce":
        workspace = ti.algorithms.ReduceWorkspace(max_items=n)
        if storage == "field":
            dst = ti.field(ti.i32, shape=())
        elif storage == "ndarray":
            dst = ti.ndarray(ti.i32, shape=1)
        else:
            payload = ti.types.struct(value=ti.i32, tag=ti.i32)
            dst_owner = ti.ndarray(payload, shape=1)
            dst_host = np.zeros((1,), dtype=dst_owner.numpy_dtype)
            dst_host["tag"] = 12345
            dst_owner.from_numpy(dst_host)
            dst = dst_owner.field("value")

        def body():
            ti.algorithms.experimental_reduce(
                src, dst, op="sum", method=method, workspace=workspace
            )

        def plan():
            return workspace._native_reduce_plan

        def verify():
            expected = np.sum(data, dtype=np.int64).astype(np.int32)
            if storage == "field":
                actual = np.int32(dst[None])
            elif storage == "ndarray":
                actual = dst.to_numpy()[0]
            else:
                result = dst_owner.to_numpy()
                if result["tag"][0] != 12345:
                    return False
                actual = result["value"][0]
            return bool(actual == expected)

        def workspace_peak():
            return max(
                int(workspace.workspace_bytes_peak),
                _runtime_workspace_peak(arch_name, op_name),
            )

        return body, plan, verify, workspace_peak

    if op_name == "transform":
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        if storage == "field":
            dst = ti.field(ti.i32, shape=n)
        elif storage == "ndarray":
            dst = ti.ndarray(ti.i32, shape=n)
        elif storage == "struct_member":
            payload = ti.types.struct(value=ti.i32, tag=ti.i32)
            dst_owner = ti.ndarray(payload, shape=n)
            dst_host = np.zeros((n,), dtype=dst_owner.numpy_dtype)
            dst_host["tag"] = np.arange(n, dtype=np.int32) * 11 - 3
            dst_owner.from_numpy(dst_host)
            dst = dst_owner.field("value")
        else:
            payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
            dst_owner = ti.ndarray(payload, shape=n)
            dst_host = np.zeros((n,), dtype=dst_owner.numpy_dtype)
            dst_host["tag"] = np.arange(n, dtype=np.int32) * 11 - 3
            dst_owner.from_numpy(dst_host)
            dst = dst_owner.field("vec")

        def body():
            ti.algorithms.experimental_transform(
                src, dst, scale=3, bias=7, method=method, workspace=workspace
            )

        def plan():
            return workspace._native_transform_plan

        def verify():
            expected = (data * np.int32(3) + np.int32(7)).astype(np.int32)
            if storage == "field":
                actual = dst.to_numpy()
            elif storage == "ndarray":
                actual = dst.to_numpy()
            elif storage == "struct_member":
                result = dst_owner.to_numpy()
                if not np.array_equal(
                    result["tag"], np.arange(n, dtype=np.int32) * 11 - 3
                ):
                    return False
                actual = result["value"]
            else:
                result = dst_owner.to_numpy()
                if not np.array_equal(
                    result["tag"], np.arange(n, dtype=np.int32) * 11 - 3
                ):
                    return False
                actual = result["vec"]
            return bool(np.array_equal(actual, expected))

        return body, plan, verify, lambda: int(workspace.workspace_bytes_peak)

    raise ValueError(op_name)


def run_child(args: argparse.Namespace) -> int:
    import taichi_forge as ti  # pylint: disable=import-outside-toplevel

    pid = os.getpid()
    sample_gpu = args.arch in ("cuda", "vulkan")
    gpu_before_init = _gpu_process_dedicated_mb(pid) if sample_gpu else None
    ti.init(arch=_arch_value(ti, args.arch), offline_cache=False, log_level=ti.ERROR)
    gpu_after_init = _gpu_process_dedicated_mb(pid) if sample_gpu else None

    available, reason = _available(ti, args.arch, args.op, args.storage)
    if not available:
        row = {
            "arch": args.arch,
            "storage": args.storage,
            "op": args.op,
            "n": args.n,
            "skipped": True,
            "skip_reason": reason,
        }
        print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
        return 0

    body, plan_getter, verify, workspace_peak = _make_body(
        ti, args.arch, args.op, args.storage, args.n
    )
    _sync(ti)
    gpu_after_alloc = _gpu_process_dedicated_mb(pid) if sample_gpu else None

    first_t0 = time.perf_counter()
    body()
    _sync(ti)
    first_ms = (time.perf_counter() - first_t0) * 1000.0
    ok = verify()
    first_plan = plan_getter()
    gpu_after_first = _gpu_process_dedicated_mb(pid) if sample_gpu else None

    for _ in range(args.warmups):
        body()
        _sync(ti)
    samples = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        body()
        _sync(ti)
        samples.append((time.perf_counter() - t0) * 1000.0)
    plan_after = plan_getter()
    gpu_after_run = _gpu_process_dedicated_mb(pid) if sample_gpu else None

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
    row = {
        "arch": args.arch,
        "storage": args.storage,
        "op": args.op,
        "n": args.n,
        "package_version": ".".join(str(x) for x in ti.__version__[:3]),
        "first_call_ms": first_ms,
        "runtime": _stats_ms(samples),
        "workspace_peak_bytes": int(workspace_peak()),
        "plan_reused": first_plan is not None and plan_after is first_plan,
        "plan_backend": None if first_plan is None else first_plan["backend"],
        "plan_method": None if first_plan is None else first_plan["method_name"],
        "ok": ok,
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
    print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
    return 0 if ok and row["plan_reused"] else 1


def _parse_child_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError("child result marker not found")


def run_matrix(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    this_script = Path(__file__).resolve()
    for arch in args.arches:
        for storage in args.storages:
            for op_name in args.ops:
                for n in args.sizes:
                    cmd = [
                        args.python,
                        str(this_script),
                        "--child",
                        "--arch",
                        arch,
                        "--storage",
                        storage,
                        "--op",
                        op_name,
                        "--n",
                        str(n),
                        "--repeats",
                        str(args.repeats),
                        "--warmups",
                        str(args.warmups),
                    ]
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["PYTHONPATH"] = args.pythonpath
                    env["TAICHI_OFFLINE_CACHE"] = "0"
                    print("RUN " + " ".join(cmd), flush=True)
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=args.timeout_s,
                    )
                    stem = f"{arch}_{storage}_{op_name}_{n}"
                    (out_dir / f"{stem}.stdout.txt").write_text(
                        proc.stdout, encoding="utf-8"
                    )
                    (out_dir / f"{stem}.stderr.txt").write_text(
                        proc.stderr, encoding="utf-8"
                    )
                    try:
                        row = _parse_child_result(proc.stdout)
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        row = None
                        failures.append(
                            {
                                "arch": arch,
                                "storage": storage,
                                "op": op_name,
                                "n": n,
                                "returncode": proc.returncode,
                                "parse_error": str(exc),
                                "stderr_tail": proc.stderr[-2000:],
                            }
                        )
                    if row is not None:
                        rows.append(row)
                        status = "SKIP" if row.get("skipped") else "OK"
                        print(
                            f"{status} {arch} {storage} {op_name} n={n} "
                            f"first={row.get('first_call_ms', 0):.3f} "
                            f"median={row.get('runtime', {}).get('median_ms', 0):.3f} "
                            f"workspace={row.get('workspace_peak_bytes', 0)}",
                            flush=True,
                        )
                    if proc.returncode != 0:
                        failures.append(
                            {
                                "arch": arch,
                                "storage": storage,
                                "op": op_name,
                                "n": n,
                                "returncode": proc.returncode,
                                "stderr_tail": proc.stderr[-2000:],
                            }
                        )
    summary = {"rows": rows, "failures": failures}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "storage",
                "op",
                "n",
                "first_call_ms",
                "runtime_median_ms",
                "workspace_peak_bytes",
                "plan_reused",
                "plan_backend",
                "plan_method",
                "gpu_peak_delta_mb",
                "ok",
                "skipped",
            ],
        )
        writer.writeheader()
        for row in rows:
            runtime = row.get("runtime", {})
            gpu = row.get("gpu_dedicated_mb", {})
            writer.writerow(
                {
                    "arch": row["arch"],
                    "storage": row["storage"],
                    "op": row["op"],
                    "n": row["n"],
                    "first_call_ms": row.get("first_call_ms"),
                    "runtime_median_ms": runtime.get("median_ms"),
                    "workspace_peak_bytes": row.get("workspace_peak_bytes"),
                    "plan_reused": row.get("plan_reused"),
                    "plan_backend": row.get("plan_backend"),
                    "plan_method": row.get("plan_method"),
                    "gpu_peak_delta_mb": gpu.get("peak_delta"),
                    "ok": row.get("ok"),
                    "skipped": row.get("skipped", False),
                }
            )
    print(f"WROTE {out_dir / 'summary.json'}")
    print(f"WROTE {out_dir / 'summary.csv'}")
    if failures:
        print(json.dumps({"failures": failures}, indent=2, ensure_ascii=False))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"])
    parser.add_argument(
        "--storage",
        choices=["field", "ndarray", "struct_member", "struct_tensor_member"],
    )
    parser.add_argument("--op", choices=["scan", "reduce", "transform"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=str(ROOT / "python"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "benchmarks" / "results" / "s4_native_plan_replay"),
    )
    parser.add_argument("--arches", nargs="+", default=["cpu", "cuda", "vulkan"])
    parser.add_argument(
        "--storages",
        nargs="+",
        default=["field", "ndarray", "struct_member", "struct_tensor_member"],
    )
    parser.add_argument("--ops", nargs="+", default=["scan", "reduce", "transform"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 1048576])
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.child:
        missing = [
            name
            for name in ("arch", "storage", "op", "n")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"missing child args: {missing}")
        return run_child(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
