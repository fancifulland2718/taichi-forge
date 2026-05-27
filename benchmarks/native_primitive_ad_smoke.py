import argparse
import collections
import collections.abc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
collections.Callable = collections.abc.Callable

import taichi_forge as ti
import taichi_forge.algorithms._autodiff as _autodiff
from taichi_forge.lang import impl


_DEBUG = os.environ.get("TI_NATIVE_AD_SMOKE_DEBUG") == "1"


def _sync():
    try:
        ti.sync()
    except Exception:
        pass


def _arch_name(arch):
    if arch == ti.cpu:
        return "cpu"
    if arch == ti.cuda:
        return "cuda"
    if arch == ti.vulkan:
        return "vulkan"
    return str(arch)


def _copy_method():
    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        return "cuda_device"
    if arch == ti.vulkan:
        return "vulkan_native"
    return "cpu_native"


def _reduce_method():
    arch = impl.current_cfg().arch
    if arch == ti.cuda:
        return "cuda_cub"
    if arch == ti.vulkan:
        return "vulkan_native"
    return "cpu_native"


def _workspace_snapshot():
    prog = impl.get_runtime().prog
    rows = {}
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
            except Exception as exc:  # pragma: no cover - diagnostic only
                rows[name] = repr(exc)
    return rows


def _run_case(name, setup, body, check, repeats):
    setup()
    _sync()
    t0 = time.perf_counter()
    body()
    _sync()
    first = (time.perf_counter() - t0) * 1000
    try:
        check()
    except AssertionError as exc:
        raise AssertionError(f"{name}: {exc}") from exc
    samples = []
    for _ in range(repeats):
        setup()
        _sync()
        t0 = time.perf_counter()
        body()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000)
        try:
            check()
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc
    return {
        "op": name,
        "first_ms": first,
        "warm_mean_ms": float(np.mean(samples)),
        "warm_min_ms": float(np.min(samples)),
        "warm_max_ms": float(np.max(samples)),
    }


def _record_count():
    tape = _autodiff.active_tape()
    if tape is None:
        return None
    return len(getattr(tape, "calls", ()))


def run_arch(arch, n, repeats):
    ti.reset()
    ti.init(arch=arch, offline_cache=False)
    rows = []

    x = ti.field(ti.f32, shape=n, needs_grad=True)
    y = ti.field(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def init_xy():
        loss[None] = 0
        loss.grad[None] = 0
        for i in range(n):
            x[i] = ti.cast(i % 17, ti.f32)
            y[i] = 0
            x.grad[i] = 0
            y.grad[i] = 0

    @ti.kernel
    def sum_y():
        for i in range(n):
            loss[None] += y[i]

    def transform_body():
        with ti.ad.Tape(loss):
            if _DEBUG:
                print("DEBUG transform active before", _autodiff.is_tape_active(), _record_count())
            ti.algorithms.experimental_transform(
                x, y, scale=2.5, bias=1.0, method=_copy_method()
            )
            if _DEBUG:
                print("DEBUG transform records after primitive", _record_count())
            sum_y()
            if _DEBUG:
                print("DEBUG transform records after sum", _record_count())

    def transform_check():
        assert abs(x.grad[0] - 2.5) < 1e-5, x.grad[0]

    rows.append(_run_case("transform", init_xy, transform_body, transform_check, repeats))

    src = ti.field(ti.f32, shape=n, needs_grad=True)
    dst = ti.field(ti.f32, shape=n, needs_grad=True)
    indices = ti.ndarray(ti.i32, shape=n)
    indices.from_numpy(np.arange(n, dtype=np.int32)[::-1].copy())

    @ti.kernel
    def init_copy():
        loss[None] = 0
        loss.grad[None] = 0
        for i in range(n):
            src[i] = ti.cast(i % 19, ti.f32)
            dst[i] = 0
            src.grad[i] = 0
            dst.grad[i] = 0

    @ti.kernel
    def sum_dst():
        for i in range(n):
            loss[None] += dst[i]

    def gather_body():
        with ti.ad.Tape(loss):
            ti.algorithms.experimental_gather(
                src, indices, dst, method=_copy_method()
            )
            sum_dst()

    def scatter_body():
        with ti.ad.Tape(loss):
            ti.algorithms.experimental_scatter(
                src, indices, dst, method=_copy_method()
            )
            sum_dst()

    def scatter_add_body():
        with ti.ad.Tape(loss):
            ti.algorithms.experimental_scatter_add(
                src, indices, dst, method=_copy_method()
            )
            sum_dst()

    def copy_check():
        assert abs(src.grad[0] - 1.0) < 1e-5, src.grad[0]

    rows.append(_run_case("gather", init_copy, gather_body, copy_check, repeats))
    rows.append(_run_case("scatter", init_copy, scatter_body, copy_check, repeats))
    rows.append(
        _run_case("scatter_add", init_copy, scatter_add_body, copy_check, repeats)
    )

    values = ti.field(ti.f32, shape=n, needs_grad=True)
    reduced = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def init_reduce():
        reduced[None] = 0
        reduced.grad[None] = 0
        for i in range(n):
            values[i] = ti.cast(i % 23, ti.f32)
            values.grad[i] = 0

    def reduce_body():
        with ti.ad.Tape(reduced):
            ti.algorithms.experimental_reduce(
                values, reduced, op="sum", method=_reduce_method()
            )

    def reduce_check():
        assert abs(values.grad[0] - 1.0) < 1e-5, values.grad[0]

    rows.append(_run_case("reduce", init_reduce, reduce_body, reduce_check, repeats))

    if arch != ti.vulkan:
        scan_values = ti.field(ti.f32, shape=n, needs_grad=True)
        scan_loss = ti.field(ti.f32, shape=(), needs_grad=True)
        scanner = ti.algorithms.PrefixSumExecutor(n)

        @ti.kernel
        def init_scan():
            scan_loss[None] = 0
            scan_loss.grad[None] = 0
            for i in range(n):
                scan_values[i] = 1
                scan_values.grad[i] = 0

        @ti.kernel
        def sum_scan():
            for i in range(n):
                scan_loss[None] += scan_values[i]

        def scan_body():
            with ti.ad.Tape(scan_loss):
                scanner.run(scan_values)
                sum_scan()

        def scan_check():
            assert abs(scan_values.grad[0] - n) < 1e-4, scan_values.grad[0]
            assert abs(scan_values.grad[n - 1] - 1) < 1e-4, scan_values.grad[n - 1]

        rows.append(_run_case("scan", init_scan, scan_body, scan_check, repeats))

    keys = ti.field(ti.i32, shape=n)
    group_values = ti.field(ti.f32, shape=n, needs_grad=True)
    grouped = ti.field(ti.f32, shape=256, needs_grad=True)
    grouped_loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def init_grouped():
        grouped_loss[None] = 0
        grouped_loss.grad[None] = 0
        for i in range(n):
            keys[i] = i % 256
            group_values[i] = 1
            group_values.grad[i] = 0
        for i in range(256):
            grouped[i] = 0
            grouped.grad[i] = 0

    @ti.kernel
    def sum_grouped():
        for i in range(256):
            grouped_loss[None] += grouped[i]

    def grouped_body():
        with ti.ad.Tape(grouped_loss):
            ti.algorithms.experimental_grouped_reduce(
                keys, group_values, grouped, op="sum", method=_copy_method()
            )
            sum_grouped()

    def grouped_check():
        assert abs(group_values.grad[0] - 1.0) < 1e-5, group_values.grad[0]

    rows.append(
        _run_case(
            "grouped_reduce", init_grouped, grouped_body, grouped_check, repeats
        )
    )
    return {
        "arch": _arch_name(arch),
        "n": n,
        "rows": rows,
        "workspace": _workspace_snapshot(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--arch",
        choices=("cpu", "cuda", "vulkan", "all"),
        default="all",
    )
    args = parser.parse_args()
    arch_map = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
    arches = arch_map.values() if args.arch == "all" else (arch_map[args.arch],)
    results = []
    for arch in arches:
        try:
            results.append(run_arch(arch, args.n, args.repeats))
        except Exception as exc:  # pragma: no cover - diagnostic only
            results.append({"arch": _arch_name(arch), "error": repr(exc)})
    print("NATIVE_AD_SMOKE " + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
