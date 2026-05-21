import argparse
import json
import statistics
import time

import numpy as np
import taichi_forge as ti
from taichi_forge.lang import impl


ARCHES = {
    "cpu": ti.cpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
}


def _method_for(arch_name, primitive):
    if arch_name == "cpu":
        return "cpu_native"
    if arch_name == "cuda":
        return "cuda_cub_native" if primitive == "sort" else "cuda_device"
    if arch_name == "vulkan":
        return "vulkan_native_radix_u32" if primitive == "sort" else "vulkan_native"
    raise ValueError(arch_name)


def _available(arch_name, primitive):
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
    if arch_name == "cuda":
        if primitive == "transform":
            return (
                hasattr(prog, "cuda_toolkit_transform_available")
                and prog.cuda_toolkit_transform_available()
            )
        if primitive == "sort":
            return (
                hasattr(prog, "cuda_cub_radix_sort_available")
                and prog.cuda_cub_radix_sort_available()
                and hasattr(prog, "cuda_toolkit_transform_available")
                and prog.cuda_toolkit_transform_available()
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
            return (
                hasattr(prog, "cuda_device_indexed_copy_available")
                and prog.cuda_device_indexed_copy_available()
            )
    if arch_name == "vulkan":
        if primitive == "transform":
            return (
                hasattr(prog, "vulkan_transform_available")
                and prog.vulkan_transform_available()
            )
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
            return (
                hasattr(prog, "vulkan_indexed_copy_available")
                and prog.vulkan_indexed_copy_available()
            )
    return False


def _payload(n):
    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
    values = ti.ndarray(payload, shape=n)
    host = np.zeros((n,), dtype=values.numpy_dtype)
    host["vec"] = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 97) - 48
    host["tag"] = np.arange(n, dtype=np.int32) * 5 + 1
    values.from_numpy(host)
    return values, host


def _time_call(fn, repeats):
    fn()
    ti.sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        ti.sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


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
    method = "cpu_native" if arch_name == "cpu" else ("cuda_cub" if arch_name == "cuda" else "vulkan_native")

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


def run_bucket(arch_name, n, repeats):
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
    method = _method_for(arch_name, "transform")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCHES), default="cpu")
    parser.add_argument("--sizes", default="2048,262144")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--primitive",
        choices=["transform", "sort", "compact", "bucket", "gather", "scatter", "all"],
        default="all",
    )
    args = parser.parse_args()

    ti.init(arch=ARCHES[args.arch], offline_cache=False)
    primitives = (
        ["transform", "gather", "scatter", "sort", "compact", "bucket"]
        if args.primitive == "all"
        else [args.primitive]
    )
    results = []
    for n in [int(item) for item in args.sizes.split(",") if item]:
        for primitive in primitives:
            if not _available(args.arch, primitive):
                results.append(
                    {
                        "arch": args.arch,
                        "n": n,
                        "primitive": primitive,
                        "ok": False,
                        "skipped": True,
                    }
                )
                continue
            if primitive == "transform":
                stats = run_transform(args.arch, n, args.repeats)
            elif primitive == "sort":
                stats = run_sort(args.arch, n, args.repeats)
            elif primitive == "compact":
                stats = run_compact(args.arch, n, args.repeats)
            elif primitive == "gather":
                stats = run_indexed_copy(args.arch, n, args.repeats, scatter=False)
            elif primitive == "scatter":
                stats = run_indexed_copy(args.arch, n, args.repeats, scatter=True)
            else:
                stats = run_bucket(args.arch, n, args.repeats)
            stats.update({"arch": args.arch, "n": n})
            results.append(stats)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
