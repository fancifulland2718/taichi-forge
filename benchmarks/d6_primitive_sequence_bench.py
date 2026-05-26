import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

import taichi_forge as ti


ROOT = Path(__file__).resolve().parents[1]


def _arch(name):
    if name == "cpu":
        return ti.cpu
    if name == "cuda":
        return ti.cuda
    if name == "vulkan":
        return ti.vulkan
    raise ValueError(name)


def _method_for(arch_name, op):
    if arch_name == "cpu":
        return "cpu_native"
    if arch_name == "cuda":
        return "cuda_device"
    if arch_name == "vulkan":
        return "vulkan_native"
    return "auto"


def _values(n):
    return ((np.arange(n, dtype=np.int32) * 17) % 251 - 113).astype(np.int32)


def _time_body(body, *, warmups, repeats):
    start = time.perf_counter()
    body()
    ti.sync()
    first_ms = (time.perf_counter() - start) * 1000.0
    for _ in range(warmups):
        body()
        ti.sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        body()
        ti.sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "first_call_ms": first_ms,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": samples,
    }


def _make_arrays(n):
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


def _check(data, dst):
    expected = (data * np.int32(3) + np.int32(5)).astype(np.int32)
    return bool(np.array_equal(dst.to_numpy(), expected))


def _run_mode(arch_name, mode, n, warmups, repeats):
    data, src, tmp, indices, gathered, dst = _make_arrays(n)
    transform_method = _method_for(arch_name, "transform")
    copy_method = _method_for(arch_name, "indexed_copy")

    if mode == "public_none":
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

        stats = _time_body(body, warmups=warmups, repeats=repeats)
        workspace_peak = 0
        direct_plans = 0
    elif mode == "explicit_workspace":
        transform_ws = ti.algorithms.TransformWorkspace(max_items=n)
        gather_ws = ti.algorithms.IndexedCopyWorkspace(max_items=n)
        scatter_ws = ti.algorithms.IndexedCopyWorkspace(max_items=n)

        def body():
            ti.algorithms.experimental_transform(
                src, tmp, scale=3, bias=5, method=transform_method, workspace=transform_ws
            )
            ti.algorithms.experimental_gather(
                tmp, indices, gathered, method=copy_method, workspace=gather_ws
            )
            ti.algorithms.experimental_scatter(
                gathered, indices, dst, method=copy_method, workspace=scatter_ws
            )

        stats = _time_body(body, warmups=warmups, repeats=repeats)
        workspace_peak = (
            transform_ws.workspace_bytes_peak
            + gather_ws.workspace_bytes_peak
            + scatter_ws.workspace_bytes_peak
        )
        direct_plans = sum(
            1
            for ws in (transform_ws, gather_ws, scatter_ws)
            if getattr(ws, "_native_transform_plan", None) is not None
            or getattr(ws, "_native_indexed_copy_plan", None) is not None
        )
    elif mode in ("primitive_sequence", "primitive_sequence_no_fusion"):
        old_fusion = os.environ.get("TAICHI_FORGE_PRIMITIVE_SEQUENCE_FUSION")
        if mode == "primitive_sequence_no_fusion":
            os.environ["TAICHI_FORGE_PRIMITIVE_SEQUENCE_FUSION"] = "0"
        try:
            seq = ti.algorithms.primitive_sequence()
            seq.transform(src, tmp, scale=3, bias=5, method=transform_method)
            seq.gather(tmp, indices, gathered, method=copy_method)
            seq.scatter(gathered, indices, dst, method=copy_method)
            start = time.perf_counter()
            seq.prewarm()
            ti.sync()
            prewarm_ms = (time.perf_counter() - start) * 1000.0
            stats = _time_body(lambda: seq.run(), warmups=warmups, repeats=repeats)
            stats["prewarm_ms"] = prewarm_ms
            stats["fused_plan_count"] = seq.fused_plan_count
            stats["fused_plan_method"] = seq.fused_plan_method
            workspace_peak = seq.workspace_bytes_peak
            direct_plans = seq.direct_plan_count
        finally:
            if old_fusion is None:
                os.environ.pop("TAICHI_FORGE_PRIMITIVE_SEQUENCE_FUSION", None)
            else:
                os.environ["TAICHI_FORGE_PRIMITIVE_SEQUENCE_FUSION"] = old_fusion
    else:
        raise ValueError(mode)

    stats.update(
        {
            "arch": arch_name,
            "mode": mode,
            "n": n,
            "warmups": warmups,
            "repeats": repeats,
            "ok": _check(data, dst),
            "workspace_peak_bytes": int(workspace_peak),
            "direct_plan_count": int(direct_plans),
        }
    )
    return stats


def _write_outputs(out_dir, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2, sort_keys=True)
    fields = [
        "arch",
        "mode",
        "n",
        "ok",
        "first_call_ms",
        "prewarm_ms",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "workspace_peak_bytes",
        "direct_plan_count",
        "fused_plan_count",
        "fused_plan_method",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arches", nargs="+", default=["cpu"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["public_none", "explicit_workspace", "primitive_sequence"],
    )
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "d6_5_primitive_sequence",
    )
    args = parser.parse_args()

    rows = []
    for arch_name in args.arches:
        for mode in args.modes:
            ti.reset()
            ti.init(arch=_arch(arch_name), offline_cache=False)
            row = _run_mode(arch_name, mode, args.n, args.warmups, args.repeats)
            rows.append(row)
            print(
                "D6_PRIMITIVE_SEQUENCE "
                + json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "arch",
                            "mode",
                            "ok",
                            "first_call_ms",
                            "prewarm_ms",
                            "median_ms",
                            "mean_ms",
                            "workspace_peak_bytes",
                            "direct_plan_count",
                            "fused_plan_count",
                            "fused_plan_method",
                        )
                    },
                    sort_keys=True,
                )
            )
    path = _write_outputs(args.out_dir, rows)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()