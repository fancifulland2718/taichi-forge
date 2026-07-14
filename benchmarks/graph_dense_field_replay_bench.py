"""Fresh-process baseline for dense Field Graph execution.

The parent process launches every direct/Graph trial in a separate process.
Use ``--matrix`` for the DF0 1K/64K/1M, 1/2/4/8/16 dispatch matrix; the
default is intentionally small enough for a local preflight.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "DENSE_FIELD_GRAPH_RESULT "


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _stats_ms(values: list[float]) -> dict[str, object]:
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "sample_ms": values,
    }


def _rss_mb() -> float | None:
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _gpu_process_mb(pid: int) -> float | None:
    if platform.system() == "Windows":
        command = (
            "$p="
            + str(pid)
            + ";$s=0;(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples|"
            + "? InstanceName -like ('pid_'+$p+'_*')|%{$s+=$_.CookedValue};"
            + "[Console]::WriteLine([math]::Round($s/1MB,3))"
        )
        argv = ["powershell", "-NoProfile", "-Command", command]
    else:
        argv = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    try:
        output = subprocess.check_output(
            argv, text=True, stderr=subprocess.DEVNULL, timeout=3.0
        )
        if platform.system() == "Windows":
            return float(output.strip())
        total = 0.0
        found = False
        for line in output.splitlines():
            process_id, used = [part.strip() for part in line.split(",", 1)]
            if int(process_id) == pid:
                total += float(used)
                found = True
        return total if found else None
    except Exception:
        return None


def _driver_version() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).splitlines()[0].strip()
    except Exception:
        return None


def _package_version() -> str | None:
    try:
        return metadata.version("taichi-forge")
    except metadata.PackageNotFoundError:
        return None


def _make_workload(ti, payload: str, size: int):
    if payload == "scalar":
        state = ti.field(ti.f32, shape=size)

        @ti.kernel
        def initialize():
            for i in state:
                state[i] = ti.cast(i % 97, ti.f32) * 0.001

        @ti.kernel
        def advance():
            for i in state:
                state[i] = state[i] * 0.99991 + ti.cast(i % 7, ti.f32) * 1e-5

        def summary():
            values = state.to_numpy()
            return {"sum": float(values.sum()), "max": float(values.max())}

        return initialize, [advance], summary, size * 4

    if payload == "vector":
        state = ti.Vector.field(3, ti.f32, shape=size)

        @ti.kernel
        def initialize():
            for i in state:
                value = ti.cast(i % 101, ti.f32) * 0.001
                state[i] = ti.Vector([value, -value, 0.5 * value])

        @ti.kernel
        def advance():
            for i in state:
                state[i] = state[i] * 0.99993 + ti.Vector([1e-5, -2e-5, 3e-5])

        def summary():
            values = state.to_numpy()
            return {"sum": float(values.sum()), "max": float(values.max())}

        return initialize, [advance], summary, size * 3 * 4

    if payload == "matrix":
        state = ti.Matrix.field(2, 2, ti.f32, shape=size)

        @ti.kernel
        def initialize():
            for i in state:
                value = ti.cast(i % 89, ti.f32) * 0.001
                state[i] = ti.Matrix([[value, 0.25], [-0.5, value + 1.0]])

        @ti.kernel
        def advance():
            for i in state:
                state[i] = state[i] * 0.99995 + ti.Matrix([[1e-5, 0.0], [0.0, -1e-5]])

        def summary():
            values = state.to_numpy()
            return {"sum": float(values.sum()), "max": float(values.max())}

        return initialize, [advance], summary, size * 4 * 4

    if payload != "production":
        raise ValueError(payload)

    position = ti.Vector.field(2, ti.f32, shape=size)
    velocity = ti.Vector.field(2, ti.f32, shape=size)
    constraint = ti.field(ti.f32, shape=size)
    snapshot = ti.Vector.field(2, ti.f32, shape=size)
    epoch = ti.field(ti.i32, shape=())

    @ti.kernel
    def initialize():
        epoch[None] = 0
        for i in position:
            value = ti.cast(i % 127, ti.f32) * 0.001
            position[i] = ti.Vector([value, 0.5 + value])
            velocity[i] = ti.Vector([0.1, -0.2])
            constraint[i] = 0.0
            snapshot[i] = ti.Vector.zero(ti.f32, 2)

    @ti.kernel
    def integrate():
        for i in position:
            velocity[i][1] -= 9.8e-4
            position[i] += velocity[i] * 1e-3

    @ti.kernel
    def project_constraint():
        for i in constraint:
            penetration = ti.max(0.0, 0.05 - position[i][1])
            constraint[i] = penetration
            position[i][1] += penetration
            velocity[i][1] += penetration * 0.1

    @ti.kernel
    def publish_snapshot():
        epoch[None] += 1
        for i in snapshot:
            snapshot[i] = position[i]

    def summary():
        values = snapshot.to_numpy()
        return {
            "sum": float(values.sum()),
            "max": float(values.max()),
            "epoch": int(epoch[None]),
        }

    payload_bytes = size * (2 * 4 * 3 + 4) + 4
    return (
        initialize,
        [integrate, project_constraint, publish_snapshot],
        summary,
        payload_bytes,
    )


def _arch_value(ti, name: str):
    return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def _core_commit() -> str | None:
    try:
        from taichi_forge._lib import core  # pylint: disable=import-outside-toplevel

        for name in ("get_commit_hash", "get_commit_hash_full"):
            function = getattr(core, name, None)
            if function is not None:
                return str(function())
    except Exception:
        pass
    return None


def _run_child(args) -> dict[str, object]:
    import_start = time.perf_counter()
    import taichi_forge as ti  # pylint: disable=import-outside-toplevel

    import_ms = (time.perf_counter() - import_start) * 1000.0
    init_start = time.perf_counter()
    try:
        ti.init(
            arch=_arch_value(ti, args.arch),
            enable_fallback=False,
            offline_cache=False,
        )
    except Exception as exc:
        return {
            "skipped": True,
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "arch": args.arch,
            "mode": args.mode,
            "payload": args.payload,
            "size": args.size,
            "dispatches": args.dispatches,
            "trial": args.trial,
        }
    init_ms = (time.perf_counter() - init_start) * 1000.0

    rss_before_mb = _rss_mb()
    sample_gpu = args.sample_gpu_memory and args.arch != "cpu"
    gpu_before_mb = _gpu_process_mb(os.getpid()) if sample_gpu else None
    setup_start = time.perf_counter()
    initialize, kernels, summarize, payload_bytes = _make_workload(
        ti, args.payload, args.size
    )
    initialize()
    ti.sync()
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    sequence = [kernels[index % len(kernels)] for index in range(args.dispatches)]
    graph = None
    graph_build_ms = 0.0
    if args.mode == "graph":
        build_start = time.perf_counter()
        builder = ti.graph.GraphBuilder()
        for kernel in sequence:
            builder.dispatch(kernel)
        graph = builder.compile()
        graph_build_ms = (time.perf_counter() - build_start) * 1000.0
        if args.diagnostics == "on":
            # Enable detailed counters outside both build and first-run timing.
            graph.execution_stats()

        def invoke():
            graph.run({})

    else:

        def invoke():
            for kernel in sequence:
                kernel()

    first_start = time.perf_counter()
    invoke()
    ti.sync()
    first_run_ms = (time.perf_counter() - first_start) * 1000.0
    rss_after_first_mb = _rss_mb()
    gpu_after_first_mb = _gpu_process_mb(os.getpid()) if sample_gpu else None

    for _ in range(args.warmups):
        for _ in range(args.batch):
            invoke()
        ti.sync()
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        for _ in range(args.batch):
            invoke()
        ti.sync()
        samples.append(
            (time.perf_counter() - start) * 1000.0 / args.batch
        )

    result = {
        "skipped": False,
        "arch": args.arch,
        "actual_arch": str(ti.lang.impl.current_cfg().arch),
        "mode": args.mode,
        "payload": args.payload,
        "size": args.size,
        "dispatches": args.dispatches,
        "trial": args.trial,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "batch": args.batch,
        "diagnostics": args.diagnostics,
        "package_file": str(Path(ti.__file__).resolve()),
        "package_version": _package_version(),
        "ti_version": list(ti.__version__),
        "runtime_commit": _core_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "driver_version": _driver_version(),
        "import_ms": import_ms,
        "init_ms": init_ms,
        "setup_ms": setup_ms,
        "graph_build_ms": graph_build_ms,
        "first_run_ms": first_run_ms,
        "field_payload_bytes": payload_bytes,
        "summary": summarize(),
        "rss_before_mb": rss_before_mb,
        "rss_after_first_mb": rss_after_first_mb,
        "rss_after_steady_mb": _rss_mb(),
        "gpu_before_mb": gpu_before_mb,
        "gpu_after_first_mb": gpu_after_first_mb,
        "gpu_after_steady_mb": (
            _gpu_process_mb(os.getpid()) if sample_gpu else None
        ),
        "graph_execution_report": (
            asdict(graph.execution_stats()) if graph is not None else None
        ),
    }
    result.update(_stats_ms(samples))
    return result


def _child_command(args, arch, mode, payload, size, dispatches, trial):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arch",
        arch,
        "--mode",
        mode,
        "--payload",
        payload,
        "--size",
        str(size),
        "--dispatches",
        str(dispatches),
        "--trial",
        str(trial),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--batch",
        str(args.batch),
        "--diagnostics",
        args.diagnostics,
    ]
    if args.sample_gpu_memory:
        command.append("--sample-gpu-memory")
    return command


def _child_env(args, arch):
    env = os.environ.copy()
    source_paths = [str(ROOT / "python"), str(ROOT)]
    if args.pythonpath:
        source_paths.insert(0, str(args.pythonpath))
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "0"
    env["TI_WANTED_ARCHS"] = arch
    return env


def _run_fresh(args, arch, mode, payload, size, dispatches, trial):
    process = subprocess.run(
        _child_command(
            args, arch, mode, payload, size, dispatches, trial
        ),
        capture_output=True,
        text=True,
        env=_child_env(args, arch),
        timeout=args.timeout,
        check=False,
    )
    if args.verbose and process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    for line in process.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            row = json.loads(line[len(RESULT_PREFIX) :])
            row["process_returncode"] = process.returncode
            return row
    raise RuntimeError(
        f"child produced no result: {arch}/{mode}/{payload}/"
        f"{size}/{dispatches}/trial-{trial}; exit={process.returncode}"
    )


def _range_percent(values: list[float]) -> float:
    median = statistics.median(values)
    return (max(values) - min(values)) / max(abs(median), 1e-12) * 100.0


def _summary_close(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        if key == "epoch":
            if left[key] != right[key]:
                return False
        elif not np.isclose(left[key], right[key], rtol=1e-5, atol=1e-6):
            return False
    return True


def _memory_delta(row: dict, prefix: str) -> float | None:
    before = row.get(f"{prefix}_before_mb")
    after = row.get(f"{prefix}_after_steady_mb")
    if before is None or after is None:
        return None
    return after - before


def _graph_paths(rows: list[dict]) -> tuple[list[str], list[str]]:
    paths = set()
    reasons = set()
    for row in rows:
        report = row.get("graph_execution_report") or {}
        for segment in report.get("segments") or []:
            if segment.get("kind") != "cgraph":
                continue
            path = segment.get("last_path")
            reason = segment.get("fallback_reason")
            if path:
                paths.add(path)
            if reason and reason != "none":
                reasons.add(reason)
    return sorted(paths), sorted(reasons)


def _aggregate(rows: list[dict]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                row["arch"],
                row["payload"],
                row["size"],
                row["dispatches"],
            )
            for row in rows
        }
    )
    aggregates = []
    for arch, payload, size, dispatches in keys:
        selected = [
            row
            for row in rows
            if (
                row["arch"],
                row["payload"],
                row["size"],
                row["dispatches"],
            )
            == (arch, payload, size, dispatches)
        ]
        direct = [row for row in selected if row["mode"] == "direct"]
        graph = [row for row in selected if row["mode"] == "graph"]
        skipped = [row for row in selected if row.get("skipped")]
        if skipped:
            aggregates.append(
                {
                    "arch": arch,
                    "payload": payload,
                    "size": size,
                    "dispatches": dispatches,
                    "skipped": True,
                    "skip_reasons": sorted(
                        {row.get("skip_reason") for row in skipped}
                    ),
                }
            )
            continue

        direct_medians = [row["median_ms"] for row in direct]
        graph_medians = [row["median_ms"] for row in graph]
        direct_steady = statistics.median(direct_medians)
        graph_steady = statistics.median(graph_medians)
        direct_first = statistics.median(
            row["first_run_ms"] for row in direct
        )
        graph_first_total = statistics.median(
            row["graph_build_ms"] + row["first_run_ms"] for row in graph
        )
        paired_ok = []
        for trial in sorted({row["trial"] for row in selected}):
            direct_row = next(row for row in direct if row["trial"] == trial)
            graph_row = next(row for row in graph if row["trial"] == trial)
            paired_ok.append(
                _summary_close(direct_row["summary"], graph_row["summary"])
            )
        paths, reasons = _graph_paths(graph)
        rss_deltas = [
            value
            for value in (_memory_delta(row, "rss") for row in graph)
            if value is not None
        ]
        gpu_deltas = [
            value
            for value in (_memory_delta(row, "gpu") for row in graph)
            if value is not None
        ]
        aggregates.append(
            {
                "arch": arch,
                "payload": payload,
                "size": size,
                "dispatches": dispatches,
                "diagnostics": rows[0]["diagnostics"],
                "skipped": False,
                "summary_ok": all(paired_ok),
                "direct_median_ms": direct_steady,
                "graph_median_ms": graph_steady,
                "steady_speedup_graph_vs_direct": direct_steady / graph_steady,
                "direct_first_ms": direct_first,
                "graph_build_plus_first_ms": graph_first_total,
                "first_speedup_graph_vs_direct": direct_first / graph_first_total,
                "direct_trial_range_pct": _range_percent(direct_medians),
                "graph_trial_range_pct": _range_percent(graph_medians),
                "confidence_ok": _range_percent(direct_medians) <= 5.0
                and _range_percent(graph_medians) <= 5.0,
                "graph_paths": paths,
                "fallback_reasons": reasons,
                "graph_rss_delta_mb": (
                    statistics.median(rss_deltas) if rss_deltas else None
                ),
                "graph_gpu_delta_mb": (
                    statistics.median(gpu_deltas) if gpu_deltas else None
                ),
            }
        )
    return aggregates


def _write_results(path: Path, rows: list[dict], aggregates: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taichi_forge.graph_dense_field.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "rows": rows,
        "aggregates": aggregates,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    csv_path = path.with_suffix(".csv")
    fieldnames = sorted({key for row in aggregates for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregates:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )
    return path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--arch", choices=["cpu", "cuda", "vulkan"], default="cpu"
    )
    parser.add_argument(
        "--archs", nargs="+", default=["cpu"], choices=["cpu", "cuda", "vulkan"]
    )
    parser.add_argument("--mode", choices=["direct", "graph"], default="direct")
    parser.add_argument(
        "--payload",
        choices=["scalar", "vector", "matrix", "production"],
        default="production",
    )
    parser.add_argument(
        "--payloads",
        nargs="+",
        default=["production"],
        choices=["scalar", "vector", "matrix", "production"],
    )
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--sizes", nargs="+", type=int, default=[65536])
    parser.add_argument("--dispatches", type=int, default=4)
    parser.add_argument("--dispatch-counts", nargs="+", type=int, default=[4])
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument(
        "--diagnostics", choices=["off", "on"], default="off"
    )
    parser.add_argument("--sample-gpu-memory", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pythonpath", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "temp_outputs"
        / "graph_dense_field"
        / "df0_baseline.json",
    )
    args = parser.parse_args()

    if args.child:
        result = _run_child(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return

    if args.matrix:
        args.payloads = ["scalar", "vector", "matrix", "production"]
        args.sizes = [1024, 65536, 1048576]
        args.dispatch_counts = [1, 2, 4, 8, 16]

    rows = []
    total = (
        len(args.archs)
        * len(args.payloads)
        * len(args.sizes)
        * len(args.dispatch_counts)
        * args.trials
        * 2
    )
    completed = 0
    for arch in args.archs:
        for payload in args.payloads:
            for size in args.sizes:
                for dispatches in args.dispatch_counts:
                    for trial in range(args.trials):
                        for mode in ("direct", "graph"):
                            completed += 1
                            print(
                                f"[{completed}/{total}] {arch} {payload} "
                                f"n={size} dispatches={dispatches} "
                                f"trial={trial} {mode}",
                                flush=True,
                            )
                            rows.append(
                                _run_fresh(
                                    args,
                                    arch,
                                    mode,
                                    payload,
                                    size,
                                    dispatches,
                                    trial,
                                )
                            )
    aggregates = _aggregate(rows)
    json_path, csv_path = _write_results(args.output, rows, aggregates)
    print("DENSE_FIELD_GRAPH_SUMMARY " + json.dumps(aggregates, sort_keys=True))
    print(f"WROTE {json_path}")
    print(f"WROTE {csv_path}")

    divergent = [
        row
        for row in aggregates
        if not row.get("skipped") and not row.get("summary_ok")
    ]
    if divergent:
        raise SystemExit("direct and Graph summaries diverged")


if __name__ == "__main__":
    main()
