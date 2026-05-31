import argparse
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "MULTIBACKEND_COMPILE_CACHE "


def _child_env(pythonpath, offline_cache):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "1" if offline_cache else "0"
    env["PYTHONPATH"] = pythonpath or str(ROOT / "python")
    return env


def _parse_arches(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _profile_sum(rows, predicate):
    total = 0.0
    calls = 0
    for row in rows:
        path = row.get("path", "")
        if not predicate(path):
            continue
        try:
            total += float(row.get("total_s", "0"))
            calls += int(row.get("calls", "0"))
        except ValueError:
            pass
    return total, calls


def _profile_totals(rows):
    def one(name, predicate):
        total, calls = _profile_sum(rows, predicate)
        return {f"{name}_s": total, f"{name}_calls": calls}

    out = {}
    out.update(one("python_frontend", lambda p: p.startswith("python.frontend.")))
    out.update(one("python_source", lambda p: p.startswith("python.frontend.") and p.endswith(".source")))
    out.update(one("python_ast_parse", lambda p: p.startswith("python.frontend.") and p.endswith(".ast_parse")))
    out.update(one("python_global_vars", lambda p: p.startswith("python.frontend.") and p.endswith(".global_vars")))
    out.update(one("python_context", lambda p: p.startswith("python.frontend.") and p.endswith(".context")))
    out.update(one("python_kernel_ast_transform", lambda p: p.startswith("python.kernel.ast_transform:")))
    out.update(one("python_func_inline_transform", lambda p: p.startswith("python.func.inline_transform:")))
    out.update(one("cpp_kernel", lambda p: "cpp.compile.kernel" in p))
    out.update(one("cpp_ir_pipeline", lambda p: "cpp.compile.ir_pipeline" in p))
    out.update(one("cpp_backend_codegen", lambda p: "cpp.compile.backend_codegen" in p))
    out.update(one("cpp_llvm_to_offloads", lambda p: "cpp.compile.llvm.compile_to_offloads" in p))
    out.update(one("cpp_spirv_to_executable", lambda p: "cpp.compile.spirv.compile_to_executable" in p))
    return out


def _cache_stats(cache_dir):
    path = Path(cache_dir)
    if not path.exists():
        return {"cache_file_count": 0, "cache_metadata_count": 0, "cache_total_bytes": 0}
    files = [item for item in path.iterdir() if item.is_file()]
    return {
        "cache_file_count": sum(1 for item in files if item.suffix == ".tic"),
        "cache_metadata_count": sum(1 for item in files if item.suffix == ".tcb"),
        "cache_total_bytes": sum(item.stat().st_size for item in files),
    }


def _make_kernels(ti, count):
    kernels = []
    def make_one(salt):
        @ti.kernel
        def kernel(
            a: ti.types.ndarray(dtype=ti.f32, ndim=1),
            b: ti.types.ndarray(dtype=ti.f32, ndim=1),
            n: ti.i32,
        ):
            for i in range(n):
                x = a[i] + ti.cast(salt, ti.f32) * 0.001
                y = x
                for _ in ti.static(range(4)):
                    y = y * 1.0001 + 0.1
                b[i] = y

        return kernel

    for salt in range(count):
        kernels.append(make_one(salt))
    return kernels


def _expected_output(host, salt):
    expected = host + np.float32(salt) * np.float32(0.001)
    for _ in range(4):
        expected = expected * np.float32(1.0001) + np.float32(0.1)
    return expected


def _run_child(args):
    import taichi_forge as ti

    arch_values = {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "opengl": ti.opengl,
    }
    kernels = _make_kernels(ti, args.kernels)
    rows = []
    for arch_name in _parse_arches(args.arches):
        arch = arch_values.get(arch_name)
        row = {"arch": arch_name, "ok": False}
        if arch is None:
            row["error"] = f"unknown arch {arch_name!r}"
            rows.append(row)
            continue
        reset_done = False
        try:
            init_kwargs = {
                "arch": arch,
                "offline_cache": bool(args.offline_cache),
                "log_level": ti.ERROR,
            }
            if args.offline_cache:
                init_kwargs["offline_cache_file_path"] = args.cache_dir
            ti.init(**init_kwargs)
            n = args.n
            host = np.linspace(0.0, 1.0, n, dtype=np.float32)
            a = ti.ndarray(ti.f32, shape=n)
            b = ti.ndarray(ti.f32, shape=n)
            a.from_numpy(host)

            t0 = time.perf_counter()
            with ti.compile_profile() as prof:
                for kernel in kernels:
                    kernel(a, b, n)
                ti.sync()
            compile_wall_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            for _ in range(args.runtime_repeats):
                for kernel in kernels:
                    kernel(a, b, n)
                ti.sync()
            runtime_s = time.perf_counter() - t1

            out = b.to_numpy()
            expected = _expected_output(host, args.kernels - 1)
            row.update(_profile_totals(prof.records()))
            row.update(
                {
                    "ok": bool(np.allclose(out, expected, rtol=1e-5, atol=1e-5)),
                    "compile_wall_s": compile_wall_s,
                    "runtime_repeat_s": runtime_s / max(args.runtime_repeats, 1),
                    "kernel_count": args.kernels,
                    "n": n,
                }
            )
            if args.offline_cache:
                ti.reset()
                reset_done = True
                row.update(_cache_stats(args.cache_dir))
        except Exception as exc:  # noqa: BLE001 - benchmark records backend availability.
            row["error"] = repr(exc)
        finally:
            try:
                if not reset_done:
                    ti.reset()
            except Exception:
                pass
        rows.append(row)
    print(RESULT_PREFIX + json.dumps(rows, sort_keys=True))


def _run_parent(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arches",
        args.arches,
        "--kernels",
        str(args.kernels),
        "--n",
        str(args.n),
        "--runtime-repeats",
        str(args.runtime_repeats),
    ]
    cache_dir = args.cache_dir or str(out_dir / "offline_cache")
    if args.offline_cache:
        cmd.extend(["--offline-cache", "--cache-dir", cache_dir])
    for repeat in range(args.repeats):
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=_child_env(args.pythonpath, args.offline_cache),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                payload = json.loads(line[len(RESULT_PREFIX) :])
                break
        if payload is None:
            payload = [
                {
                    "arch": "unknown",
                    "ok": False,
                    "error": "missing benchmark payload",
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-2000:],
                    "stderr_tail": proc.stderr[-2000:],
                }
            ]
        for row in payload:
            row["repeat"] = repeat
            rows.append(row)

    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Multi-backend Compile Cache Benchmark",
        "",
        "| repeat | arch | ok | compile ms | frontend ms | ast parse ms | transform ms | cpp ir ms | backend ms | runtime repeat ms | cache files | metadata |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {repeat} | {arch} | {ok} | {compile_ms:.3f} | {frontend_ms:.3f} | {ast_ms:.3f} | {transform_ms:.3f} | {ir_ms:.3f} | {backend_ms:.3f} | {runtime_ms:.3f} | {cache_files} | {metadata} |".format(
                repeat=row.get("repeat", 0),
                arch=row.get("arch", ""),
                ok=int(bool(row.get("ok"))),
                compile_ms=1000.0 * float(row.get("compile_wall_s", 0.0)),
                frontend_ms=1000.0 * float(row.get("python_frontend_s", 0.0)),
                ast_ms=1000.0 * float(row.get("python_ast_parse_s", 0.0)),
                transform_ms=1000.0 * float(row.get("python_kernel_ast_transform_s", 0.0)),
                ir_ms=1000.0 * float(row.get("cpp_ir_pipeline_s", 0.0)),
                backend_ms=1000.0 * float(row.get("cpp_backend_codegen_s", 0.0)),
                runtime_ms=1000.0 * float(row.get("runtime_repeat_s", 0.0)),
                cache_files=int(row.get("cache_file_count", 0)),
                metadata=int(row.get("cache_metadata_count", 0)),
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--arches", default="cpu,cuda,vulkan")
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--runtime-repeats", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results" / "multibackend_compile_cache"))
    parser.add_argument("--pythonpath", default=None)
    parser.add_argument("--offline-cache", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    if args.offline_cache and args.cache_dir is None:
        args.cache_dir = str(ROOT / "benchmarks" / "results" / "multibackend_compile_cache" / "offline_cache")
    if args.child:
        _run_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
