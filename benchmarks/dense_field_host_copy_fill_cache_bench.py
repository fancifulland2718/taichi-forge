from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "DENSE_FIELD_HOST_COPY_FILL_CACHE "


def _install_descriptor_patch():
    field_mod = importlib.import_module("taichi_forge.lang.field")
    impl = importlib.import_module("taichi_forge.lang.impl")
    misc = importlib.import_module("taichi_forge.lang.misc")
    scalar_cls = field_mod.ScalarField
    to_numpy_type = field_mod.to_numpy_type
    value_type_of = field_mod._dense_host_copy_value_type
    fill_bits = field_mod._dense_fill_value_bits
    missing = object()
    descriptor_cache = {}

    def descriptor(prog, name):
        cached = descriptor_cache.get(name, missing)
        if cached is missing:
            cached = getattr(type(prog), name, None)
            descriptor_cache[name] = cached
        return cached

    def try_from_numpy(self, arr):
        if not hasattr(arr, "flags") or not arr.flags.c_contiguous:
            return False
        if impl.current_cfg().arch not in (misc.x64, misc.arm64):
            return False
        value_type = value_type_of(self.dtype)
        if value_type is None:
            return False
        if arr.dtype != to_numpy_type(self.dtype):
            return False
        prog = impl.get_runtime().prog
        method = descriptor(prog, "copy_dense_field_from_host")
        if method is None:
            return False
        impl.get_runtime().materialize()
        try:
            method(prog, self.snode.ptr, arr, value_type, int(arr.size))
        except RuntimeError as exc:
            message = str(exc)
            if "CPU native dense field host copy" not in message and "Native dense field path" not in message:
                raise
            return False
        return True

    def try_to_numpy(self, arr):
        if not hasattr(arr, "flags") or not arr.flags.c_contiguous:
            return False
        if impl.current_cfg().arch not in (misc.x64, misc.arm64):
            return False
        value_type = value_type_of(self.dtype)
        if value_type is None:
            return False
        if arr.dtype != to_numpy_type(self.dtype):
            return False
        prog = impl.get_runtime().prog
        method = descriptor(prog, "copy_dense_field_to_host")
        if method is None:
            return False
        impl.get_runtime().materialize()
        try:
            method(prog, self.snode.ptr, arr, value_type, int(arr.size))
        except RuntimeError as exc:
            message = str(exc)
            if "CPU native dense field host readback" not in message and "Native dense field path" not in message:
                raise
            return False
        return True

    def try_fill(self, val):
        if impl.current_cfg().arch not in (misc.x64, misc.arm64):
            return False
        value_type = value_type_of(self.dtype)
        if value_type is None:
            return False
        value_bits = fill_bits(self.dtype, val)
        if value_bits is None:
            return False
        prog = impl.get_runtime().prog
        method = descriptor(prog, "fill_dense_field")
        if method is None:
            return False
        impl.get_runtime().materialize()
        n = int(np.prod(self.shape, dtype=np.int64))
        try:
            method(prog, self.snode.ptr, value_type, value_bits, n)
        except RuntimeError as exc:
            message = str(exc)
            if "CPU native dense field fill" not in message and "Native dense field path" not in message:
                raise
            return False
        return True

    scalar_cls._try_cpu_dense_from_numpy = try_from_numpy
    scalar_cls._try_cpu_dense_to_numpy = try_to_numpy
    scalar_cls._try_cpu_dense_fill = try_fill


def _stats(samples):
    return {
        "samples": len(samples),
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "stdev_us": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }

def run_child(args) -> int:
    if args.mode == "descriptor":
        _install_descriptor_patch()
    import taichi_forge as ti  # pylint: disable=import-outside-toplevel

    ti.init(arch=ti.cpu, offline_cache=False)
    x = ti.field(ti.i32, shape=args.n)
    arr = ((np.arange(args.n, dtype=np.int32) * 17 + 5) % 1009).astype(np.int32)
    x.from_numpy(arr)
    ti.sync()

    if args.op == "fill":
        def body():
            x.fill(123)
    elif args.op == "from_numpy":
        def body():
            x.from_numpy(arr)
    elif args.op == "to_numpy":
        def body():
            x.to_numpy()
    else:
        raise ValueError(args.op)

    for _ in range(args.warmups):
        body()
        ti.sync()
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        body()
        ti.sync()
        samples.append((time.perf_counter() - start) * 1_000_000.0)
    if args.op == "fill":
        ok = bool(np.all(x.to_numpy() == 123))
    elif args.op == "from_numpy":
        ok = bool(np.array_equal(x.to_numpy(), arr))
    else:
        ok = bool(np.array_equal(x.to_numpy(), arr))
    row = {
        "mode": args.mode,
        "op": args.op,
        "n": args.n,
        "ok": ok,
        "warmups": args.warmups,
        **_stats(samples),
    }
    print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
    return 0 if ok else 1


def _child_env(pythonpath):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "0"
    env["PYTHONPATH"] = pythonpath or str(ROOT / "python")
    return env


def _parse_child(stdout):
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise ValueError("missing child result")


def _repeats_for(n, base):
    if base is not None:
        return int(base)
    if n <= 16:
        return 5000
    if n <= 1024:
        return 2000
    if n <= 65536:
        return 300
    return 80


def run_matrix(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    script = Path(__file__).resolve()
    for n in args.sizes:
        repeats = _repeats_for(n, args.repeats)
        for op in args.ops:
            for trial in range(args.trials):
                for mode in ("baseline", "descriptor"):
                    cmd = [
                        args.python,
                        str(script),
                        "--child",
                        "--mode",
                        mode,
                        "--op",
                        op,
                        "--n",
                        str(n),
                        "--repeats",
                        str(repeats),
                        "--warmups",
                        str(args.warmups),
                    ]
                    stem = f"{mode}_{op}_{n}_trial{trial}"
                    print("RUN " + " ".join(cmd), flush=True)
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        env=_child_env(args.pythonpath),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=args.timeout_s,
                    )
                    (out_dir / f"{stem}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
                    (out_dir / f"{stem}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
                    if proc.returncode != 0:
                        failures.append({"mode": mode, "op": op, "n": n, "trial": trial, "returncode": proc.returncode, "stderr_tail": proc.stderr[-2000:]})
                        print(f"FAIL {mode} {op} n={n} trial={trial}", flush=True)
                        continue
                    row = _parse_child(proc.stdout)
                    row["trial"] = trial
                    row["repeats"] = repeats
                    rows.append(row)
                    print(f"OK {mode} {op} n={n} trial={trial} median={row['median_us']:.3f}us mean={row['mean_us']:.3f}us", flush=True)
    summary = {"rows": rows, "failures": failures, "sizes": args.sizes, "ops": args.ops, "trials": args.trials, "warmups": args.warmups}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "op", "n", "trial", "repeats", "warmups", "ok", "samples", "median_us", "mean_us", "min_us", "max_us", "stdev_us"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {out_dir / 'summary.json'}")
    print(f"WROTE {csv_path}")
    return 1 if failures else 0

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=["baseline", "descriptor"], default="baseline")
    parser.add_argument("--op", choices=["fill", "from_numpy", "to_numpy"], default="fill")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=str(ROOT / "python"))
    parser.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results" / "dense_field_host_copy_fill_cache"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 16, 1024, 65536, 1048576])
    parser.add_argument("--ops", nargs="+", default=["fill", "from_numpy", "to_numpy"], choices=["fill", "from_numpy", "to_numpy"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.child:
        if args.repeats is None:
            args.repeats = _repeats_for(args.n, None)
        return run_child(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())