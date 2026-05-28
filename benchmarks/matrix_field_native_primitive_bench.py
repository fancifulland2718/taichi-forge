from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np


def _arch_value(ti, name):
    if name == "cpu":
        return ti.cpu
    if name == "cuda":
        return ti.cuda
    if name == "vulkan":
        return ti.vulkan
    raise ValueError(name)


def _method_for(ti, name):
    if name == "cpu":
        return "cpu_native"
    if name == "cuda":
        return "cuda_device"
    if name == "vulkan":
        return "vulkan_native"
    raise ValueError(name)


def _reduce_method_for(name):
    return "cuda_cub" if name == "cuda" else _method_for(None, name)


def _sync(ti):
    ti.sync()


def _time_call(ti, fn, repeats):
    start = time.perf_counter()
    fn()
    _sync(ti)
    first_ms = (time.perf_counter() - start) * 1000.0
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(ti)
        samples.append((time.perf_counter() - start) * 1000.0)
    return first_ms, statistics.fmean(samples), statistics.median(samples)


def _component_fields(field):
    return tuple(field.get_scalar_field(i) for i in range(field.n))


def _workspace_method(workspace, attr):
    plan = getattr(workspace, attr, None)
    if plan is not None:
        return getattr(plan, "method_name", "")
    plans = getattr(workspace, attr + "s", None)
    if plans:
        return ",".join(sorted({p.method_name for p in plans.values()}))
    return ""


def _component_plan_method(workspace, attr):
    group = getattr(workspace, attr, None)
    if group is None:
        return ""
    return ",".join(sorted({call[1] for call in group.stage_calls}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["cpu", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    import taichi_forge as ti

    ti.init(arch=_arch_value(ti, args.arch), offline_cache=True)

    n = args.n
    groups = args.groups
    method = _method_for(ti, args.arch)
    reduce_method = _reduce_method_for(args.arch)
    values_np = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 17) - 8
    dst_np = np.zeros_like(values_np)
    indices_np = (n - 1 - np.arange(n, dtype=np.int32)).astype(np.int32)
    keys_np = ((np.arange(n, dtype=np.int32) * 31 + 5) % groups).astype(np.int32)
    group_base = np.zeros((groups, 2), dtype=np.int32)

    src = ti.Vector.field(2, ti.i32, shape=n)
    dst = ti.Vector.field(2, ti.i32, shape=n)
    reduce_out = ti.Vector.field(2, ti.i32, shape=())
    group_out = ti.Vector.field(2, ti.i32, shape=groups)
    indices_field = ti.field(ti.i32, shape=n)
    keys_field = ti.field(ti.i32, shape=n)
    keys_nd = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(values_np)
    dst.from_numpy(dst_np)
    indices_field.from_numpy(indices_np)
    keys_field.from_numpy(keys_np)
    keys_nd.from_numpy(keys_np)
    group_out.from_numpy(group_base)

    results = []

    def record(op, mode, fn, workspace, method_attr, component_attr=None):
        try:
            first, mean, median = _time_call(ti, fn, args.repeats)
            workspace_obj = workspace() if callable(workspace) else workspace
            method_name = (
                _component_plan_method(workspace_obj, component_attr)
                if component_attr
                else _workspace_method(workspace_obj, method_attr)
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 - benchmark records unsupported paths.
            workspace_obj = workspace() if callable(workspace) else workspace
            first = mean = median = None
            method_name = ""
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "arch": args.arch,
                "op": op,
                "mode": mode,
                "first_ms": first,
                "warm_mean_ms": mean,
                "warm_median_ms": median,
                "workspace_bytes_peak": getattr(
                    workspace_obj, "workspace_bytes_peak", 0
                ),
                "method": method_name,
                "error": error,
            }
        )

    reduce_ws = ti.algorithms.ReduceWorkspace(max_items=n)

    def reduce_packed():
        ti.algorithms.experimental_reduce(
            src, reduce_out, method=reduce_method, workspace=reduce_ws
        )

    record("reduce", "packed", reduce_packed, reduce_ws, "_native_reduce_plan")

    reduce_component_ws = ti.algorithms.ReduceWorkspace(max_items=n)
    src_components = _component_fields(src)
    out_components = _component_fields(reduce_out)

    def reduce_component():
        for src_c, out_c in zip(src_components, out_components):
            ti.algorithms.experimental_reduce(
                src_c,
                out_c,
                method=reduce_method,
                workspace=reduce_component_ws,
            )

    record(
        "reduce",
        "component",
        reduce_component,
        reduce_component_ws,
        "_native_reduce_plan",
    )

    gather_ws = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    def gather_packed_field_indices():
        ti.algorithms.experimental_gather(
            src, indices_field, dst, method=method, workspace=gather_ws
        )

    record(
        "gather_field_indices",
        "packed",
        gather_packed_field_indices,
        gather_ws,
        "_native_indexed_copy_plan",
    )

    gather_component_ws = ti.algorithms.IndexedCopyWorkspace(max_items=n)
    dst_components = _component_fields(dst)

    def gather_component_field_indices():
        for src_c, dst_c in zip(src_components, dst_components):
            ti.algorithms.experimental_gather(
                src_c,
                indices_field,
                dst_c,
                method=method,
                workspace=gather_component_ws,
            )

    record(
        "gather_field_indices",
        "component",
        gather_component_field_indices,
        gather_component_ws,
        "_native_indexed_copy_plan",
    )

    scatter_ws = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    def scatter_packed_field_indices():
        ti.algorithms.experimental_scatter(
            src, indices_field, dst, method=method, workspace=scatter_ws
        )

    record(
        "scatter_field_indices",
        "packed",
        scatter_packed_field_indices,
        scatter_ws,
        "_native_indexed_copy_plan",
    )

    scatter_component_ws = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    def scatter_component_field_indices():
        for src_c, dst_c in zip(src_components, dst_components):
            ti.algorithms.experimental_scatter(
                src_c,
                indices_field,
                dst_c,
                method=method,
                workspace=scatter_component_ws,
            )

    record(
        "scatter_field_indices",
        "component",
        scatter_component_field_indices,
        scatter_component_ws,
        "_native_indexed_copy_plan",
    )

    scatter_add_ws = ti.algorithms.ScatterAddWorkspace(max_items=n)

    def scatter_add_packed_field_indices():
        group_out.fill(0)
        ti.algorithms.experimental_scatter_add(
            src, keys_field, group_out, method=method, workspace=scatter_add_ws
        )

    record(
        "scatter_add_field_indices",
        "packed",
        scatter_add_packed_field_indices,
        scatter_add_ws,
        "_native_scatter_add_plan",
    )

    scatter_add_component_ws = ti.algorithms.ScatterAddWorkspace(max_items=n)
    group_components = _component_fields(group_out)

    def scatter_add_component_field_indices():
        group_out.fill(0)
        for src_c, out_c in zip(src_components, group_components):
            ti.algorithms.experimental_scatter_add(
                src_c,
                keys_field,
                out_c,
                method=method,
                workspace=scatter_add_component_ws,
            )

    record(
        "scatter_add_field_indices",
        "component",
        scatter_add_component_field_indices,
        scatter_add_component_ws,
        "_native_scatter_add_plan",
    )

    grouped_ws = ti.algorithms.GroupedReduceWorkspace(max_items=n, max_groups=groups)

    def grouped_reduce_packed_ndarray_keys():
        ti.algorithms.experimental_grouped_reduce(
            keys_nd, src, group_out, method=method, workspace=grouped_ws
        )

    record(
        "grouped_reduce_ndarray_keys",
        "packed",
        grouped_reduce_packed_ndarray_keys,
        grouped_ws,
        "_native_grouped_reduce_plan",
        component_attr="_packed_grouped_reduce_plan_group",
    )

    grouped_component_ws = ti.algorithms.GroupedReduceWorkspace(
        max_items=n, max_groups=groups
    )

    def grouped_reduce_component_ndarray_keys():
        for src_c, out_c in zip(src_components, group_components):
            ti.algorithms.experimental_grouped_reduce(
                keys_nd,
                src_c,
                out_c,
                method=method,
                workspace=grouped_component_ws,
            )

    record(
        "grouped_reduce_ndarray_keys",
        "component",
        grouped_reduce_component_ndarray_keys,
        grouped_component_ws,
        "_native_grouped_reduce_plan",
    )

    grouped_field_ws = ti.algorithms.GroupedReduceWorkspace(
        max_items=n, max_groups=groups
    )

    def grouped_reduce_packed_field_keys():
        ti.algorithms.experimental_grouped_reduce(
            keys_field, src, group_out, method=method, workspace=grouped_field_ws
        )

    record(
        "grouped_reduce_field_keys",
        "packed",
        grouped_reduce_packed_field_keys,
        grouped_field_ws,
        "_native_grouped_reduce_plan",
        component_attr="_packed_grouped_reduce_plan_group",
    )

    grouped_field_component_ws = ti.algorithms.GroupedReduceWorkspace(
        max_items=n, max_groups=groups
    )

    def grouped_reduce_component_field_keys():
        for src_c, out_c in zip(src_components, group_components):
            ti.algorithms.experimental_grouped_reduce(
                keys_field,
                src_c,
                out_c,
                method=method,
                workspace=grouped_field_component_ws,
            )

    record(
        "grouped_reduce_field_keys",
        "component",
        grouped_reduce_component_field_keys,
        grouped_field_component_ws,
        "_native_grouped_reduce_plan",
    )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
