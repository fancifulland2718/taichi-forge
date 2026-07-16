import argparse
import json
import statistics
import time

import numpy as np
import taichi_forge as ti
from taichi_forge.lang import impl

try:
    from benchmarks.gpu_idle_guard import (
        finalize_performance_measurement,
        prepare_performance_measurement,
    )
except ModuleNotFoundError:
    from gpu_idle_guard import (
        finalize_performance_measurement,
        prepare_performance_measurement,
    )


ARCHES = {
    "cpu": ti.cpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
}
_METHOD_MODE = "native"
_WARMUPS = 3
_INTERNAL_STATS = False


def _method_for(arch_name, primitive):
    if _METHOD_MODE == "auto":
        return "auto"
    if arch_name == "cpu":
        return "cpu_native"
    if arch_name == "cuda":
        if primitive == "sort":
            return "cuda_cub_native"
        if primitive == "compact":
            return "cuda_cub"
        return "cuda_device"
    if arch_name == "vulkan":
        return "vulkan_native_radix_u32" if primitive == "sort" else "vulkan_native"
    raise ValueError(arch_name)


def _available(arch_name, primitive, method=None):
    prog = impl.get_runtime().prog
    if arch_name == "cpu":
        if primitive == "transform":
            return hasattr(prog, "cpu_transform_available") and prog.cpu_transform_available()
        if primitive == "sort":
            return hasattr(prog, "cpu_stable_sort_available") and prog.cpu_stable_sort_available()
        if primitive == "compact":
            return (
                hasattr(prog, "cpu_compact_available")
                and prog.cpu_compact_available()
                and hasattr(prog, "cpu_indexed_copy_available")
                and prog.cpu_indexed_copy_available()
            )
        if primitive == "bucket":
            return (
                hasattr(prog, "cpu_bucket_builder_available")
                and prog.cpu_bucket_builder_available()
                and hasattr(prog, "cpu_indexed_copy_available")
                and prog.cpu_indexed_copy_available()
            )
        if primitive in ("gather", "scatter"):
            return hasattr(prog, "cpu_indexed_copy_available") and prog.cpu_indexed_copy_available()
        if primitive == "scan":
            return hasattr(prog, "cpu_scan_available") and prog.cpu_scan_available()
        if primitive == "reduce":
            return hasattr(prog, "cpu_reduce_available") and prog.cpu_reduce_available()
        if primitive == "scatter_add":
            if method in ("two_level", "cpu_two_level"):
                return (
                    hasattr(prog, "cpu_grouped_reduce_available")
                    and prog.cpu_grouped_reduce_available()
                    and hasattr(prog, "cpu_add_merge_available")
                    and prog.cpu_add_merge_available()
                )
            return hasattr(prog, "cpu_scatter_add_available") and prog.cpu_scatter_add_available()
        if primitive == "grouped_reduce":
            return hasattr(prog, "cpu_grouped_reduce_available") and prog.cpu_grouped_reduce_available()
        if primitive == "histogram":
            return hasattr(prog, "cpu_histogram_available") and prog.cpu_histogram_available()
    if arch_name == "cuda":
        if primitive == "transform":
            return hasattr(prog, "cuda_device_transform_available") and prog.cuda_device_transform_available()
        if primitive == "sort":
            return (
                hasattr(prog, "cuda_cub_radix_sort_available")
                and prog.cuda_cub_radix_sort_available()
                and hasattr(prog, "cuda_device_transform_available")
                and prog.cuda_device_transform_available()
                and hasattr(prog, "cuda_device_indexed_copy_available")
                and prog.cuda_device_indexed_copy_available()
            )
        if primitive == "compact":
            return (
                hasattr(prog, "cuda_cub_select_available")
                and prog.cuda_cub_select_available()
                and hasattr(prog, "cuda_device_indexed_copy_available")
                and prog.cuda_device_indexed_copy_available()
                and hasattr(prog, "cuda_toolkit_transform_available")
                and prog.cuda_toolkit_transform_available()
            )
        if primitive == "bucket":
            return (
                hasattr(prog, "cuda_device_bucket_builder_available")
                and prog.cuda_device_bucket_builder_available()
                and hasattr(prog, "cuda_device_indexed_copy_available")
                and prog.cuda_device_indexed_copy_available()
            )
        if primitive in ("gather", "scatter"):
            return hasattr(prog, "cuda_device_indexed_copy_available") and prog.cuda_device_indexed_copy_available()
        if primitive == "scan":
            return hasattr(prog, "cuda_device_scan_available") and prog.cuda_device_scan_available()
        if primitive == "reduce":
            return hasattr(prog, "cuda_device_reduce_available") and prog.cuda_device_reduce_available()
        if primitive == "scatter_add":
            if method in ("two_level", "cuda_two_level"):
                return (
                    hasattr(prog, "cuda_device_grouped_reduce_available")
                    and prog.cuda_device_grouped_reduce_available()
                    and hasattr(prog, "cuda_device_add_merge_available")
                    and prog.cuda_device_add_merge_available()
                )
            return hasattr(prog, "cuda_device_scatter_add_available") and prog.cuda_device_scatter_add_available()
        if primitive == "grouped_reduce":
            return hasattr(prog, "cuda_device_grouped_reduce_available") and prog.cuda_device_grouped_reduce_available()
        if primitive == "histogram":
            return hasattr(prog, "cuda_device_histogram_available") and prog.cuda_device_histogram_available()
    if arch_name == "vulkan":
        if primitive == "transform":
            return hasattr(prog, "vulkan_transform_available") and prog.vulkan_transform_available()
        if primitive == "sort":
            return (
                hasattr(prog, "vulkan_radix_sort_available")
                and prog.vulkan_radix_sort_available()
                and hasattr(prog, "vulkan_indexed_copy_available")
                and prog.vulkan_indexed_copy_available()
                and hasattr(prog, "vulkan_transform_available")
                and prog.vulkan_transform_available()
            )
        if primitive == "compact":
            return (
                hasattr(prog, "vulkan_compact_available")
                and prog.vulkan_compact_available()
                and hasattr(prog, "vulkan_indexed_copy_available")
                and prog.vulkan_indexed_copy_available()
            )
        if primitive == "bucket":
            return (
                hasattr(prog, "vulkan_bucket_builder_available")
                and prog.vulkan_bucket_builder_available()
                and hasattr(prog, "vulkan_indexed_copy_available")
                and prog.vulkan_indexed_copy_available()
            )
        if primitive in ("gather", "scatter"):
            return hasattr(prog, "vulkan_indexed_copy_available") and prog.vulkan_indexed_copy_available()
        if primitive == "scan":
            return hasattr(prog, "vulkan_scan_available") and prog.vulkan_scan_available()
        if primitive == "reduce":
            return hasattr(prog, "vulkan_reduce_available") and prog.vulkan_reduce_available()
        if primitive == "scatter_add":
            if method in ("two_level", "vulkan_two_level"):
                return (
                    hasattr(prog, "vulkan_grouped_reduce_available")
                    and prog.vulkan_grouped_reduce_available()
                    and hasattr(prog, "vulkan_add_merge_available")
                    and prog.vulkan_add_merge_available()
                    and hasattr(prog, "vulkan_transform_available")
                    and prog.vulkan_transform_available()
                )
            return hasattr(prog, "vulkan_scatter_add_available") and prog.vulkan_scatter_add_available()
        if primitive == "grouped_reduce":
            return hasattr(prog, "vulkan_grouped_reduce_available") and prog.vulkan_grouped_reduce_available()
        if primitive == "histogram":
            return (
                hasattr(prog, "vulkan_histogram_available")
                and prog.vulkan_histogram_available()
                and hasattr(prog, "vulkan_transform_available")
                and prog.vulkan_transform_available()
            )
    return False


def _runtime_workspace_peak(arch_name, primitive):
    prog = impl.get_runtime().prog
    candidates = []
    if primitive == "scan":
        candidates = [
            f"{arch_name}_scan_workspace_bytes",
            "cuda_device_scan_workspace_bytes",
            "cuda_cub_scan_workspace_bytes",
        ]
    elif primitive == "reduce":
        candidates = [
            f"{arch_name}_reduce_workspace_bytes",
            "cuda_device_reduce_workspace_bytes",
            "cuda_cub_reduce_workspace_bytes",
        ]
    for name in candidates:
        if hasattr(prog, name):
            return getattr(prog, name)()
    return 0


def _enable_internal_stats():
    if not _INTERNAL_STATS:
        return
    if hasattr(ti.algorithms, "set_primitive_diagnostics_enabled"):
        ti.algorithms.set_primitive_diagnostics_enabled(True, clear=True)
    if hasattr(ti.algorithms, "set_legacy_helper_fallback_counting_enabled"):
        ti.algorithms.set_legacy_helper_fallback_counting_enabled(True, clear=True)
    if hasattr(impl, "set_sync_diagnostics_enabled"):
        impl.set_sync_diagnostics_enabled(True, clear=True)


def _legacy_counts_for_json(counts):
    result = {}
    for key, value in counts.items():
        if isinstance(key, tuple):
            key = "|".join(str(item) for item in key)
        result[str(key)] = value
    return result


def _collect_internal_stats(expected_sync_calls):
    if not _INTERNAL_STATS:
        return None
    primitive = {}
    legacy = {}
    sync = {}
    if hasattr(ti.algorithms, "get_primitive_diagnostics"):
        primitive = ti.algorithms.get_primitive_diagnostics(reset=True)
    if hasattr(ti.algorithms, "get_legacy_helper_fallback_counts"):
        legacy = _legacy_counts_for_json(
            ti.algorithms.get_legacy_helper_fallback_counts(reset=True)
        )
    if hasattr(impl, "get_sync_diagnostics"):
        sync = impl.get_sync_diagnostics(reset=True)
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
def _payload(n):
    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
    values = ti.ndarray(payload, shape=n)
    host = np.zeros((n,), dtype=values.numpy_dtype)
    host["vec"] = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 97) - 48
    host["tag"] = np.arange(n, dtype=np.int32) * 5 + 1
    values.from_numpy(host)
    return values, host


def _time_call(fn, repeats):
    _enable_internal_stats()
    expected_sync_calls = 1 + _WARMUPS + repeats
    start = time.perf_counter()
    fn()
    ti.sync()
    first_call_ms = (time.perf_counter() - start) * 1000.0
    for _ in range(_WARMUPS):
        fn()
        ti.sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        ti.sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    result = {
        "first_call_ms": first_call_ms,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "warmups": _WARMUPS,
    }
    internal = _collect_internal_stats(expected_sync_calls)
    if internal is not None:
        result["internal"] = internal
    return result


def _sorted_rows(values):
    flat = values.reshape(values.shape[0], -1)
    if flat.shape[0] == 0:
        return flat
    order = np.lexsort(tuple(flat[:, col] for col in range(flat.shape[1] - 1, -1, -1)))
    return flat[order]


def run_transform(arch_name, n, repeats):
    values, host = _payload(n)
    out = ti.ndarray(values.struct_type, shape=n)
    out.from_numpy(host)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    method = _method_for(arch_name, "transform")

    def body():
        ti.algorithms.experimental_transform(
            values.field("vec"),
            out.field("vec"),
            scale=3,
            bias=5,
            method=method,
            workspace=workspace,
        )

    stats = _time_call(body, repeats)
    result = out.to_numpy()
    expected = (host["vec"] * np.int32(3) + np.int32(5)).astype(np.int32)
    stats.update(
        {
            "primitive": "transform",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(
                np.array_equal(result["vec"], expected)
                and np.array_equal(result["tag"], host["tag"])
            ),
        }
    )
    return stats


def run_sort(arch_name, n, repeats):
    values, host = _payload(n)
    keys_np = ((np.arange(n, dtype=np.int32) * 37) % max(n, 1)) - (n // 2)
    keys = ti.ndarray(ti.i32, shape=n)
    keys.from_numpy(keys_np)
    workspace = ti.algorithms.SortWorkspace(max_items=n)
    method = _method_for(arch_name, "sort")

    def body():
        ti.algorithms.sort(keys, values.field("vec"), method=method, workspace=workspace)

    stats = _time_call(body, repeats)
    order = np.argsort(keys_np, kind="stable")
    result = values.to_numpy()
    stats.update(
        {
            "primitive": "sort_tensor_member_values",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(
                np.array_equal(keys.to_numpy(), keys_np[order])
                and np.array_equal(result["vec"], host["vec"][order])
                and np.array_equal(result["tag"], host["tag"])
            ),
        }
    )
    return stats


def run_compact(arch_name, n, repeats):
    values, host = _payload(n)
    flags_np = ((np.arange(n) % 3 == 0) | (np.arange(n) % 17 == 0)).astype(np.int32)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(values.struct_type, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    flags.from_numpy(flags_np)
    output.fill(0)
    count.fill(0)
    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    method = _method_for(arch_name, "compact")

    def body():
        ti.algorithms.experimental_compact(
            values.field("vec"),
            flags,
            output.field("vec"),
            count,
            method=method,
            workspace=workspace,
        )

    stats = _time_call(body, repeats)
    result = output.to_numpy()
    selected = flags_np != 0
    selected_count = int(np.count_nonzero(selected))
    stats.update(
        {
            "primitive": "compact_tensor_member_values",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(
                count.to_numpy()[0] == selected_count
                and np.array_equal(result["vec"][:selected_count], host["vec"][selected])
            ),
        }
    )
    return stats


def run_bucket(arch_name, n, repeats, method_override=None):
    values, host = _payload(n)
    num_bins = max(8, min(257, n // 16))
    keys_np = ((np.arange(n, dtype=np.int32) * 37 + 11) % num_bins).astype(np.int32)
    if n >= 8:
        keys_np[1] = -1
        keys_np[5] = num_bins + 3
    counts = np.bincount(keys_np[(keys_np >= 0) & (keys_np < num_bins)], minlength=num_bins)
    expected_offsets = np.zeros(num_bins + 1, dtype=np.int32)
    expected_offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
    keys = ti.ndarray(ti.i32, shape=n)
    offsets = ti.ndarray(ti.i32, shape=num_bins + 1)
    output = ti.ndarray(values.struct_type, shape=n)
    keys.from_numpy(keys_np)
    offsets.fill(0)
    output.fill(0)
    workspace = ti.algorithms.BucketBuilderWorkspace(max_items=n, max_bins=num_bins)
    method = method_override or _method_for(arch_name, "transform")

    def body():
        ti.algorithms.experimental_bucket_builder(
            keys,
            values.field("vec"),
            offsets,
            output.field("vec"),
            method=method,
            workspace=workspace,
        )

    stats = _time_call(body, repeats)
    result_offsets = offsets.to_numpy()
    result = output.to_numpy()["vec"]
    ok = np.array_equal(result_offsets, expected_offsets)
    if ok:
        for bucket in range(num_bins):
            begin = expected_offsets[bucket]
            end = expected_offsets[bucket + 1]
            lhs = _sorted_rows(result[begin:end])
            rhs = _sorted_rows(host["vec"][keys_np == bucket])
            if not np.array_equal(lhs, rhs):
                ok = False
                break
    stats.update(
        {
            "primitive": "bucket_tensor_member_values",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(ok),
        }
    )
    return stats


def run_indexed_copy(arch_name, n, repeats, scatter):
    values, host = _payload(n)
    output = ti.ndarray(values.struct_type, shape=n)
    output.fill(0)
    indices_np = ((np.arange(n, dtype=np.int32) * 17 + 5) % max(n, 1)).astype(np.int32)
    indices = ti.ndarray(ti.i32, shape=n)
    indices.from_numpy(indices_np)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)
    method = _method_for(arch_name, "transform")

    def body():
        if scatter:
            ti.algorithms.experimental_scatter(
                values.field("vec"),
                indices,
                output.field("vec"),
                method=method,
                workspace=workspace,
            )
        else:
            ti.algorithms.experimental_gather(
                values.field("vec"),
                indices,
                output.field("vec"),
                method=method,
                workspace=workspace,
            )

    stats = _time_call(body, repeats)
    result = output.to_numpy()
    if scatter:
        expected = np.zeros_like(host["vec"])
        expected[indices_np] = host["vec"]
    else:
        expected = host["vec"][indices_np]
    stats.update(
        {
            "primitive": "scatter_tensor_member_values" if scatter else "gather_tensor_member_values",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(np.array_equal(result["vec"], expected)),
        }
    )
    return stats


def run_scan(arch_name, n, repeats):
    values, host = _payload(n)
    executor = ti.algorithms.PrefixSumExecutor(n)

    def body():
        executor.run(values.field("vec"))

    stats = _time_call(body, repeats)
    check_values, _ = _payload(n)
    ti.algorithms.PrefixSumExecutor(n).run(check_values.field("vec"))
    result = check_values.to_numpy()
    expected = np.cumsum(host["vec"], axis=0, dtype=np.int64).astype(np.int32)
    stats.update(
        {
            "primitive": "scan_tensor_member_values",
            "workspace_peak": _runtime_workspace_peak(arch_name, "scan"),
            "ok": bool(
                np.array_equal(result["vec"], expected)
                and np.array_equal(result["tag"], host["tag"])
            ),
        }
    )
    return stats


def run_reduce(arch_name, n, repeats):
    values, host = _payload(n)
    output = ti.ndarray(values.struct_type, shape=1)
    out_host = np.zeros((1,), dtype=output.numpy_dtype)
    out_host["tag"] = 12345
    output.from_numpy(out_host)
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    method = _method_for(arch_name, "reduce")

    def body():
        ti.algorithms.experimental_reduce(
            values.field("vec"),
            output.field("vec"),
            op="sum",
            method=method,
            workspace=workspace,
        )

    stats = _time_call(body, repeats)
    check_output = ti.ndarray(values.struct_type, shape=1)
    check_output.from_numpy(out_host)
    ti.algorithms.experimental_reduce(
        values.field("vec"),
        check_output.field("vec"),
        op="sum",
        method=method,
        workspace=workspace,
    )
    result = check_output.to_numpy()
    expected = np.sum(host["vec"], axis=0, dtype=np.int64).astype(np.int32)
    stats.update(
        {
            "primitive": "reduce_tensor_member_values",
            "workspace_peak": max(
                workspace.workspace_bytes_peak,
                _runtime_workspace_peak(arch_name, "reduce"),
            ),
            "ok": bool(
                np.array_equal(result["vec"][0], expected)
                and result["tag"][0] == out_host["tag"][0]
            ),
        }
    )
    return stats


def run_scatter_add(arch_name, n, repeats, method_override=None):
    values, host = _payload(n)
    buckets = max(8, min(257, n // 16))
    indices_np = ((np.arange(n, dtype=np.int32) * 37 + 11) % buckets).astype(np.int32)
    base_vec = (np.arange(buckets * 2, dtype=np.int32).reshape(buckets, 2) % 5) - 2
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(values.struct_type, shape=buckets)
    indices.from_numpy(indices_np)
    dst_host = np.zeros((buckets,), dtype=dst.numpy_dtype)
    dst_host["vec"] = base_vec
    dst_host["tag"] = np.arange(buckets, dtype=np.int32) * 11 - 5
    dst.from_numpy(dst_host)
    workspace = ti.algorithms.ScatterAddWorkspace(max_items=n)
    method = method_override or _method_for(arch_name, "scatter_add")

    def body():
        ti.algorithms.experimental_scatter_add(
            values.field("vec"),
            indices,
            dst.field("vec"),
            method=method,
            workspace=workspace,
        )

    stats = _time_call(body, repeats)
    check_dst = ti.ndarray(values.struct_type, shape=buckets)
    check_dst.from_numpy(dst_host)
    ti.algorithms.experimental_scatter_add(
        values.field("vec"),
        indices,
        check_dst.field("vec"),
        method=method,
        workspace=workspace,
    )
    expected = base_vec.copy()
    np.add.at(expected, indices_np, host["vec"])
    result = check_dst.to_numpy()
    stats.update(
        {
            "primitive": "scatter_add_tensor_member_values",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(
                np.array_equal(result["vec"], expected)
                and np.array_equal(result["tag"], dst_host["tag"])
            ),
        }
    )
    return stats


def run_grouped_reduce(arch_name, n, repeats, method_override=None):
    values, host = _payload(n)
    groups = max(8, min(257, n // 16))
    keys_np = ((np.arange(n, dtype=np.int32) * 37 + 11) % groups).astype(np.int32)
    keys = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(values.struct_type, shape=groups)
    keys.from_numpy(keys_np)
    out_host = np.zeros((groups,), dtype=output.numpy_dtype)
    out_host["vec"] = -999
    out_host["tag"] = np.arange(groups, dtype=np.int32) * 13 + 9
    output.from_numpy(out_host)
    workspace = ti.algorithms.GroupedReduceWorkspace(max_items=n, max_groups=groups)
    method = method_override or _method_for(arch_name, "grouped_reduce")

    def body():
        ti.algorithms.experimental_grouped_reduce(
            keys,
            values.field("vec"),
            output.field("vec"),
            method=method,
            workspace=workspace,
        )

    stats = _time_call(body, repeats)
    check_output = ti.ndarray(values.struct_type, shape=groups)
    check_output.from_numpy(out_host)
    ti.algorithms.experimental_grouped_reduce(
        keys,
        values.field("vec"),
        check_output.field("vec"),
        method=method,
        workspace=workspace,
    )
    expected = np.zeros((groups, 2), dtype=np.int32)
    np.add.at(expected, keys_np, host["vec"])
    result = check_output.to_numpy()
    stats.update(
        {
            "primitive": "grouped_reduce_tensor_member_values",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(
                np.array_equal(result["vec"], expected)
                and np.array_equal(result["tag"], out_host["tag"])
            ),
        }
    )
    return stats


def run_histogram(arch_name, n, repeats, method_override=None):
    num_bins = max(8, min(257, n // 16))
    payload = ti.types.struct(value=ti.i32, tag=ti.i32)
    bin_payload = ti.types.struct(count=ti.i32, tag=ti.i32)
    values = ti.ndarray(payload, shape=n)
    bins = ti.ndarray(bin_payload, shape=num_bins)
    values_np = ((np.arange(n, dtype=np.int32) * 37 + 11) % num_bins).astype(np.int32)
    values_host = np.zeros((n,), dtype=values.numpy_dtype)
    values_host["value"] = values_np
    values_host["tag"] = np.arange(n, dtype=np.int32) * 5 + 1
    bins_host = np.zeros((num_bins,), dtype=bins.numpy_dtype)
    bins_host["count"] = -1
    bins_host["tag"] = np.arange(num_bins, dtype=np.int32) * 13 + 9
    values.from_numpy(values_host)
    bins.from_numpy(bins_host)
    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    method = method_override or _method_for(arch_name, "histogram")

    def body():
        ti.algorithms.experimental_histogram(
            values.field("value"),
            bins.field("count"),
            method=method,
            workspace=workspace,
        )

    stats = _time_call(body, repeats)
    check_bins = ti.ndarray(bin_payload, shape=num_bins)
    check_bins.from_numpy(bins_host)
    ti.algorithms.experimental_histogram(
        values.field("value"),
        check_bins.field("count"),
        method=method,
        workspace=workspace,
    )
    expected = np.bincount(values_np, minlength=num_bins).astype(np.int32)
    result = check_bins.to_numpy()
    stats.update(
        {
            "primitive": "histogram_scalar_member_values",
            "workspace_peak": workspace.workspace_bytes_peak,
            "ok": bool(
                np.array_equal(result["count"], expected)
                and np.array_equal(result["tag"], bins_host["tag"])
            ),
        }
    )
    return stats


def main():
    global _METHOD_MODE, _WARMUPS, _INTERNAL_STATS  # pylint: disable=global-statement

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCHES), default="cpu")
    parser.add_argument("--sizes", default="2048,262144")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bucket-method", default=None)
    parser.add_argument("--grouped-reduce-method", default=None)
    parser.add_argument("--histogram-method", default=None)
    parser.add_argument("--scatter-add-method", default=None)
    parser.add_argument("--method-mode", choices=["native", "auto"], default="native")
    parser.add_argument("--internal-stats", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument(
        "--primitive",
        choices=[
            "transform",
            "scan",
            "reduce",
            "sort",
            "compact",
            "bucket",
            "gather",
            "scatter",
            "scatter_add",
            "grouped_reduce",
            "histogram",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()
    _METHOD_MODE = args.method_mode
    _WARMUPS = args.warmups
    _INTERNAL_STATS = args.internal_stats

    measurement_context = prepare_performance_measurement(
        args.arch,
        requested=args.performance,
    )
    ti.init(arch=ARCHES[args.arch], offline_cache=False)
    primitives = (
        [
            "transform",
            "scan",
            "reduce",
            "gather",
            "scatter",
            "scatter_add",
            "grouped_reduce",
            "histogram",
            "sort",
            "compact",
            "bucket",
        ]
        if args.primitive == "all"
        else [args.primitive]
    )
    results = []
    for n in [int(item) for item in args.sizes.split(",") if item]:
        for primitive in primitives:
            method_override = None
            if primitive == "scatter_add":
                method_override = args.scatter_add_method
            elif primitive == "grouped_reduce":
                method_override = args.grouped_reduce_method
            elif primitive == "histogram":
                method_override = args.histogram_method
            elif primitive == "bucket":
                method_override = args.bucket_method
            if not _available(args.arch, primitive, method_override):
                results.append(
                    {
                        "arch": args.arch,
                        "n": n,
                        "primitive": primitive,
                        "ok": False,
                        "skipped": True,
                        **finalize_performance_measurement(
                            measurement_context,
                            correct=False,
                            skipped=True,
                            reason="provider unavailable",
                        ),
                    }
                )
                continue
            if primitive == "transform":
                stats = run_transform(args.arch, n, args.repeats)
            elif primitive == "scan":
                stats = run_scan(args.arch, n, args.repeats)
            elif primitive == "reduce":
                stats = run_reduce(args.arch, n, args.repeats)
            elif primitive == "sort":
                stats = run_sort(args.arch, n, args.repeats)
            elif primitive == "compact":
                stats = run_compact(args.arch, n, args.repeats)
            elif primitive == "gather":
                stats = run_indexed_copy(args.arch, n, args.repeats, scatter=False)
            elif primitive == "scatter":
                stats = run_indexed_copy(args.arch, n, args.repeats, scatter=True)
            elif primitive == "scatter_add":
                stats = run_scatter_add(
                    args.arch, n, args.repeats, args.scatter_add_method
                )
            elif primitive == "grouped_reduce":
                stats = run_grouped_reduce(
                    args.arch, n, args.repeats, args.grouped_reduce_method
                )
            elif primitive == "histogram":
                stats = run_histogram(
                    args.arch, n, args.repeats, args.histogram_method
                )
            else:
                stats = run_bucket(args.arch, n, args.repeats, args.bucket_method)
            stats.update(
                {
                    "arch": args.arch,
                    "n": n,
                    **finalize_performance_measurement(
                        measurement_context,
                        correct=bool(stats.get("ok")),
                    ),
                }
            )
            results.append(stats)
    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
    print(payload)


if __name__ == "__main__":
    main()
