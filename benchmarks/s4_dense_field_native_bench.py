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

try:
    from benchmarks.gpu_idle_guard import prepare_performance_measurement
except ModuleNotFoundError:
    from gpu_idle_guard import prepare_performance_measurement


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "S4_DENSE_FIELD_BENCH "
_BUCKET_OVERRIDE: int | None = None
_KEY_PATTERN = "default"


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
        return "cuda_cub" if op_name in ("reduce", "histogram") else "cuda_device"
    if arch_name == "vulkan":
        return "vulkan_native"
    raise ValueError(arch_name)


def _values(n: int):
    import numpy as np  # pylint: disable=import-outside-toplevel

    return (np.arange(n, dtype=np.int32) % 17 - 8).astype(np.int32)


def _indices(n: int):
    import numpy as np  # pylint: disable=import-outside-toplevel

    return (n - 1 - np.arange(n, dtype=np.int32)).astype(np.int32)


def _bucket_count(n: int) -> int:
    if _BUCKET_OVERRIDE is not None:
        return max(1, min(n, _BUCKET_OVERRIDE))
    return max(64, min(n, 4096))


def _bucket_indices(n: int):
    import numpy as np  # pylint: disable=import-outside-toplevel

    buckets = _bucket_count(n)
    if _KEY_PATTERN == "single":
        return np.zeros(n, dtype=np.int32)
    if _KEY_PATTERN == "hot":
        hot_buckets = max(1, min(buckets, 64))
        return (np.arange(n, dtype=np.int32) % hot_buckets).astype(np.int32)
    return ((np.arange(n, dtype=np.int32) * 13 + 5) % buckets).astype(np.int32)


def _compact_flags(n: int):
    import numpy as np  # pylint: disable=import-outside-toplevel

    values = np.arange(n, dtype=np.int32)
    return ((values % 5 == 0) | (values % 7 == 0)).astype(np.int32)


def _histogram_values(n: int):
    import numpy as np  # pylint: disable=import-outside-toplevel

    buckets = _bucket_count(n)
    if _KEY_PATTERN == "single":
        return np.zeros(n, dtype=np.int32)
    if _KEY_PATTERN == "hot":
        hot_buckets = max(1, min(buckets, 64))
        return (np.arange(n, dtype=np.int32) % hot_buckets).astype(np.int32)
    return ((np.arange(n, dtype=np.int32) * 7 + 3) % buckets).astype(np.int32)


def _make_forge_indices(ti, n: int, values, storage: str):
    if storage == "ndarray":
        indices = ti.ndarray(ti.i32, shape=n)
    elif storage == "field":
        indices = ti.field(ti.i32, shape=n)
    else:
        raise ValueError(storage)
    indices.from_numpy(values)
    return indices


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


def _enable_internal_stats(ti, args: argparse.Namespace) -> None:
    if not getattr(args, "internal_stats", False) or args.package != "forge":
        return
    algorithms = getattr(ti, "algorithms", None)
    impl_mod = getattr(getattr(ti, "lang", None), "impl", None)
    if algorithms is not None and hasattr(algorithms, "set_primitive_diagnostics_enabled"):
        algorithms.set_primitive_diagnostics_enabled(True, clear=True)
    if algorithms is not None and hasattr(algorithms, "set_legacy_helper_fallback_counting_enabled"):
        algorithms.set_legacy_helper_fallback_counting_enabled(True, clear=True)
    if impl_mod is not None and hasattr(impl_mod, "set_sync_diagnostics_enabled"):
        impl_mod.set_sync_diagnostics_enabled(True, clear=True)


def _legacy_counts_for_json(counts: dict) -> dict:
    result = {}
    for key, value in counts.items():
        if isinstance(key, tuple):
            key = "|".join(str(item) for item in key)
        result[str(key)] = value
    return result


def _collect_internal_stats(ti, args: argparse.Namespace, expected_sync_calls: int):
    if not getattr(args, "internal_stats", False) or args.package != "forge":
        return None
    algorithms = getattr(ti, "algorithms", None)
    impl_mod = getattr(getattr(ti, "lang", None), "impl", None)
    primitive = {}
    legacy = {}
    sync = {}
    if algorithms is not None and hasattr(algorithms, "get_primitive_diagnostics"):
        primitive = algorithms.get_primitive_diagnostics(reset=True)
    if algorithms is not None and hasattr(algorithms, "get_legacy_helper_fallback_counts"):
        legacy = _legacy_counts_for_json(
            algorithms.get_legacy_helper_fallback_counts(reset=True)
        )
    if impl_mod is not None and hasattr(impl_mod, "get_sync_diagnostics"):
        sync = impl_mod.get_sync_diagnostics(reset=True)
    sync_count = int(sync.get("count", 0) or 0)
    return {
        "sync": {
            "count": sync_count,
            "total_ms": float(sync.get("total_ms", 0.0) or 0.0),
            "expected_benchmark_calls": int(expected_sync_calls),
            "extra_calls": max(0, sync_count - int(expected_sync_calls)),
        },
        "primitive": primitive,
        "legacy_helper_fallbacks": legacy,
    }
def _make_forge_body(
    ti,
    arch_name: str,
    op_name: str,
    n: int,
    method_override: str | None = None,
    indices_storage: str = "ndarray",
):
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
        method = method_override or _method_for(arch_name, op_name)

        def body():
            ti.algorithms.experimental_reduce(
                src, dst, op="sum", method=method, workspace=workspace
            )

        return body, {"workspace": workspace}

    if op_name == "transform":
        dst = ti.field(ti.i32, shape=n)
        workspace = ti.algorithms.TransformWorkspace(max_items=n)
        method = method_override or _method_for(arch_name, op_name)

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

    if op_name in ("gather", "scatter"):
        indices = _make_forge_indices(ti, n, _indices(n), indices_storage)
        dst = ti.field(ti.i32, shape=n)
        workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)
        method = method_override or _method_for(arch_name, op_name)

        if op_name == "gather":

            def body():
                ti.algorithms.experimental_gather(
                    src, indices, dst, method=method, workspace=workspace
                )

        else:

            def body():
                ti.algorithms.experimental_scatter(
                    src, indices, dst, method=method, workspace=workspace
                )

        return body, {"workspace": workspace}

    if op_name == "scatter_add":
        buckets = _bucket_count(n)
        indices = _make_forge_indices(ti, n, _bucket_indices(n), indices_storage)
        dst = ti.field(ti.i32, shape=buckets)
        workspace = ti.algorithms.ScatterAddWorkspace(max_items=n)
        method = method_override or _method_for(arch_name, op_name)

        def body():
            ti.algorithms.experimental_scatter_add(
                src, indices, dst, method=method, workspace=workspace
            )

        return body, {"workspace": workspace, "aux_size": buckets}

    if op_name == "compact":
        flags = ti.field(ti.i32, shape=n)
        output = ti.field(ti.i32, shape=n)
        count = ti.field(ti.i32, shape=())
        workspace = ti.algorithms.CompactWorkspace(max_items=n)
        flags.from_numpy(_compact_flags(n))
        method = method_override or "auto"

        def body():
            ti.algorithms.experimental_compact(
                src, flags, output, count, method=method, workspace=workspace
            )

        return body, {"workspace": workspace}

    if op_name == "histogram":
        buckets = _bucket_count(n)
        src.from_numpy(_histogram_values(n))
        bins = ti.field(ti.i32, shape=buckets)
        workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=buckets)
        method = method_override or _method_for(arch_name, op_name)

        def body():
            ti.algorithms.experimental_histogram(
                src, bins, method=method, workspace=workspace
            )

        return body, {"workspace": workspace, "aux_size": buckets}

    if op_name == "bucket_builder":
        buckets = _bucket_count(n)
        keys = ti.field(ti.i32, shape=n)
        offsets = ti.field(ti.i32, shape=buckets + 1)
        output = ti.field(ti.i32, shape=n)
        workspace = ti.algorithms.BucketBuilderWorkspace(
            max_items=n, max_bins=buckets
        )
        method = method_override or _method_for(arch_name, op_name)
        keys.from_numpy(_bucket_indices(n))

        def body():
            ti.algorithms.experimental_bucket_builder(
                keys, src, offsets, output, method=method, workspace=workspace
            )

        return body, {"workspace": workspace, "aux_size": buckets}

    if op_name == "grouped_reduce":
        buckets = _bucket_count(n)
        keys = ti.field(ti.i32, shape=n)
        output = ti.field(ti.i32, shape=buckets)
        workspace = ti.algorithms.GroupedReduceWorkspace(
            max_items=n, max_groups=buckets
        )
        method = method_override or _method_for(arch_name, op_name)
        keys.from_numpy(_bucket_indices(n))

        def body():
            ti.algorithms.experimental_grouped_reduce(
                keys, src, output, method=method, workspace=workspace
            )

        return body, {"workspace": workspace, "aux_size": buckets}

    if op_name == "sort":
        keys = ti.field(ti.i32, shape=n)
        keys.from_numpy(_bucket_indices(n))
        workspace = ti.algorithms.SortWorkspace(max_items=n)
        method = method_override or "auto"

        def body():
            ti.algorithms.sort(keys, src, method=method, workspace=workspace)

        return body, {"workspace": workspace}

    raise ValueError(op_name)


def _make_vanilla_body(ti, arch_name: str, op_name: str, n: int):
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

    if op_name in ("gather", "scatter"):
        indices = ti.field(ti.i32, shape=n)
        dst = ti.field(ti.i32, shape=n)
        indices.from_numpy(_indices(n))

        if op_name == "gather":

            @ti.kernel
            def gather_field():
                for i in indices:
                    dst[i] = src[indices[i]]

            return gather_field, {}

        @ti.kernel
        def scatter_field():
            for i in indices:
                dst[indices[i]] = src[i]

        return scatter_field, {}

    if op_name == "scatter_add":
        buckets = _bucket_count(n)
        indices = ti.field(ti.i32, shape=n)
        dst = ti.field(ti.i32, shape=buckets)
        indices.from_numpy(_bucket_indices(n))

        @ti.kernel
        def scatter_add_field():
            for i in indices:
                idx = indices[i]
                if 0 <= idx < buckets:
                    ti.atomic_add(dst[idx], src[i])

        return scatter_add_field, {"aux_size": buckets}

    if op_name == "compact":
        flags = ti.field(ti.i32, shape=n)
        output = ti.field(ti.i32, shape=n)
        count = ti.field(ti.i32, shape=())
        flags.from_numpy(_compact_flags(n))

        if arch_name == "cpu":
            @ti.kernel
            def compact_field():
                count[None] = 0
                ti.loop_config(serialize=True)
                for i in range(n):
                    if flags[i] != 0:
                        output[count[None]] = src[i]
                        count[None] += 1

            return compact_field, {}

        prefix = ti.field(ti.i32, shape=n)
        scanner = ti.algorithms.PrefixSumExecutor(n)

        @ti.kernel
        def flags_to_prefix():
            for i in src:
                prefix[i] = 1 if flags[i] != 0 else 0

        @ti.kernel
        def compact_field():
            count[None] = prefix[n - 1] if n > 0 else 0
            for i in src:
                if flags[i] != 0:
                    output[prefix[i] - 1] = src[i]

        def stable_compact_field():
            flags_to_prefix()
            scanner.run(prefix)
            compact_field()

        return stable_compact_field, {"workspace_peak_bytes": n * 4}

    if op_name == "histogram":
        buckets = _bucket_count(n)
        src.from_numpy(_histogram_values(n))
        bins = ti.field(ti.i32, shape=buckets)

        @ti.kernel
        def histogram_field():
            for i in range(buckets):
                bins[i] = 0
            for i in src:
                value = src[i]
                if 0 <= value < buckets:
                    ti.atomic_add(bins[value], 1)

        return histogram_field, {"aux_size": buckets}

    if op_name == "bucket_builder":
        buckets = _bucket_count(n)
        keys = ti.field(ti.i32, shape=n)
        offsets = ti.field(ti.i32, shape=buckets + 1)
        output = ti.field(ti.i32, shape=n)
        cursor = ti.field(ti.i32, shape=buckets)
        keys.from_numpy(_bucket_indices(n))

        @ti.kernel
        def bucket_count():
            for i in range(buckets + 1):
                offsets[i] = 0
            for i in keys:
                key = keys[i]
                if 0 <= key < buckets:
                    ti.atomic_add(offsets[key + 1], 1)

        @ti.kernel
        def bucket_prefix_serial():
            ti.loop_config(serialize=True)
            for i in range(buckets):
                offsets[i + 1] += offsets[i]

        @ti.kernel
        def bucket_copy_cursor():
            for i in range(buckets):
                cursor[i] = offsets[i]

        @ti.kernel
        def bucket_scatter():
            for i in keys:
                key = keys[i]
                if 0 <= key < buckets:
                    pos = ti.atomic_add(cursor[key], 1)
                    output[pos] = src[i]

        def bucket_builder_field():
            bucket_count()
            bucket_prefix_serial()
            bucket_copy_cursor()
            bucket_scatter()

        return bucket_builder_field, {
            "aux_size": buckets,
            "workspace_peak_bytes": buckets * 4,
        }

    if op_name == "grouped_reduce":
        buckets = _bucket_count(n)
        keys = ti.field(ti.i32, shape=n)
        output = ti.field(ti.i32, shape=buckets)
        keys.from_numpy(_bucket_indices(n))

        @ti.kernel
        def grouped_reduce_field():
            for i in range(buckets):
                output[i] = 0
            for i in keys:
                key = keys[i]
                if 0 <= key < buckets:
                    ti.atomic_add(output[key], src[i])

        return grouped_reduce_field, {"aux_size": buckets}

    if op_name == "sort":
        keys = ti.field(ti.i32, shape=n)
        keys.from_numpy(_bucket_indices(n))
        sort_fn = getattr(ti.algorithms, "sort", None)
        if sort_fn is None:
            sort_fn = ti.algorithms.parallel_sort

        def body():
            sort_fn(keys, src)

        return body, {}

    raise ValueError(op_name)


def run_child(args: argparse.Namespace) -> int:
    global _BUCKET_OVERRIDE, _KEY_PATTERN  # pylint: disable=global-statement

    _BUCKET_OVERRIDE = args.bucket_override
    _KEY_PATTERN = args.key_pattern
    measurement_context = prepare_performance_measurement(
        args.arch,
        requested=args.performance,
    )
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
        body, meta = _make_forge_body(
            ti,
            args.arch,
            args.op,
            args.n,
            args.method_override,
            args.indices_storage,
        )
    else:
        body, meta = _make_vanilla_body(ti, args.arch, args.op, args.n)
    _sync(ti)
    gpu_after_alloc = _powershell_gpu_process_dedicated_mb(pid) if sample_gpu else None

    _enable_internal_stats(ti, args)

    first_t0 = time.perf_counter()
    body()
    _sync(ti)
    first_ms = (time.perf_counter() - first_t0) * 1000.0
    gpu_after_first = _powershell_gpu_process_dedicated_mb(pid) if sample_gpu else None

    expected_sync_calls = 1 + args.warmups + args.repeats
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

    workspace_peak = int(meta.get("workspace_peak_bytes", 0) or 0)
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

    internal = _collect_internal_stats(ti, args, expected_sync_calls)

    result = {
        **measurement_context,
        "package": args.package,
        "package_version": ".".join(str(x) for x in ti.__version__[:3]),
        "arch": args.arch,
        "op": args.op,
        "method_override": args.method_override,
        "indices_storage": args.indices_storage,
        "bucket_override": args.bucket_override,
        "key_pattern": args.key_pattern,
        "dtype": "i32",
        "n": args.n,
        "aux_size": meta.get("aux_size"),
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
    if internal is not None:
        result["internal"] = internal
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
                    if args.method_override and package == "forge":
                        cmd.extend(["--method-override", args.method_override])
                    if args.indices_storage != "ndarray" and package == "forge":
                        cmd.extend(["--indices-storage", args.indices_storage])
                    if args.bucket_override is not None:
                        cmd.extend(["--bucket-override", str(args.bucket_override)])
                    if args.internal_stats:
                        cmd.append("--internal-stats")
                    if args.performance:
                        cmd.append("--performance")
                    if args.key_pattern != "default":
                        cmd.extend(["--key-pattern", args.key_pattern])
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

    summary = {
        "method_override": args.method_override,
        "indices_storage": args.indices_storage,
        "rows": rows,
        "failures": failures,
        "skips": skips,
    }
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
                "method_override",
                "indices_storage",
                "bucket_override",
                "key_pattern",
                "dtype",
                "n",
                "aux_size",
                "first_call_ms",
                "runtime_median_ms",
                "runtime_mean_ms",
                "workspace_peak_bytes",
                "gpu_peak_delta_mb",
                "gpu_peak_mb",
                "init_ms",
                "performance_valid",
                "gpu_idle_verified",
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
                    "method_override": row.get("method_override"),
                    "indices_storage": row.get("indices_storage"),
                    "bucket_override": row.get("bucket_override"),
                    "key_pattern": row.get("key_pattern"),
                    "dtype": row["dtype"],
                    "n": row["n"],
                    "aux_size": row.get("aux_size"),
                    "first_call_ms": row["first_call_ms"],
                    "runtime_median_ms": row["runtime"]["median_ms"],
                    "runtime_mean_ms": row["runtime"]["mean_ms"],
                    "workspace_peak_bytes": row["workspace_peak_bytes"],
                    "gpu_peak_delta_mb": row["gpu_dedicated_mb"]["peak_delta"],
                    "gpu_peak_mb": row["gpu_dedicated_mb"]["peak"],
                    "init_ms": row["init_ms"],
                    "performance_valid": row["measurement"][
                        "performance_valid"
                    ],
                    "gpu_idle_verified": (
                        row["measurement"].get("gpu_idle") or {}
                    ).get("verified", False),
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
    parser.add_argument(
        "--op",
        choices=[
            "scan",
            "reduce",
            "transform",
            "gather",
            "scatter",
            "scatter_add",
            "compact",
            "histogram",
            "bucket_builder",
            "grouped_reduce",
            "sort",
        ],
    )
    parser.add_argument("--n", type=int)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--method-override")
    parser.add_argument("--internal-stats", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--indices-storage", choices=["ndarray", "field"], default="ndarray")
    parser.add_argument("--bucket-override", type=int)
    parser.add_argument(
        "--key-pattern", choices=["default", "hot", "single"], default="default"
    )
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
