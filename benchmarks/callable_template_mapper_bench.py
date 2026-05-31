import argparse
import json
import time
from pathlib import Path

import numpy as np
import taichi_forge as ti


def _measure(name, mapper, args, repeats):
    mapper.lookup(args)
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        mapper.lookup(args)
    elapsed_ns = time.perf_counter_ns() - t0
    return {
        "case": name,
        "lookup_repeats": repeats,
        "mapping_size": len(mapper.mapping),
        "num_args": len(args),
        "us_per_lookup": elapsed_ns / repeats / 1000.0,
    }


def _make_cases():
    x = ti.field(ti.i32, shape=())
    y = ti.field(ti.i32, shape=())

    arr_a = ti.ndarray(ti.f32, shape=1024)
    arr_b = ti.ndarray(ti.f32, shape=1024)
    host = np.zeros((16, 8), dtype=np.float32)

    @ti.kernel
    def primitive_kernel(a: ti.i32, b: ti.f32, c: ti.i64, d: ti.f64):
        x[None] = a + ti.cast(b, ti.i32) + ti.cast(c, ti.i32) + ti.cast(d, ti.i32)

    @ti.kernel
    def ndarray_kernel(
        a: ti.types.ndarray(dtype=ti.f32, ndim=1),
        b: ti.types.ndarray(dtype=ti.f32, ndim=1),
        n: ti.i32,
    ):
        for i in range(n):
            b[i] = a[i]

    @ti.kernel
    def external_array_kernel(a: ti.types.ndarray(dtype=ti.f32, ndim=2), n: ti.i32):
        x[None] = n + ti.cast(a.shape[0], ti.i32)

    @ti.kernel
    def template_kernel(a: ti.template(), b: ti.template(), c: ti.i32):
        a[None] = b[None] + c

    @ti.kernel
    def mixed_kernel(
        a: ti.i32,
        b: ti.f32,
        c: ti.types.ndarray(dtype=ti.f32, ndim=1),
        d: ti.i32,
        e: ti.template(),
        f: ti.i64,
    ):
        e[None] = a + d + ti.cast(f, ti.i32) + ti.cast(b, ti.i32) + ti.cast(c.shape[0], ti.i32)

    return [
        ("primitive", primitive_kernel._primal.mapper, (1, 2.0, 3, 4.0)),
        ("ndarray", ndarray_kernel._primal.mapper, (arr_a, arr_b, 1024)),
        ("external_array", external_array_kernel._primal.mapper, (host, 16)),
        ("template", template_kernel._primal.mapper, (x, y, 3)),
        ("mixed", mixed_kernel._primal.mapper, (1, 2.0, arr_a, 3, x, 4)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup-repeats", type=int, default=200000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", default="benchmarks/results/callable_template_mapper_bench")
    args = parser.parse_args()

    ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
    rows = []
    for repeat in range(args.repeats):
        for name, mapper, call_args in _make_cases():
            row = _measure(name, mapper, call_args, args.lookup_repeats)
            row["repeat"] = repeat
            rows.append(row)
    ti.reset()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Callable Template Mapper Benchmark",
        "",
        "| repeat | case | args | mapping | us/lookup |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {repeat} | {case} | {num_args} | {mapping_size} | {us_per_lookup:.3f} |".format(**row)
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
