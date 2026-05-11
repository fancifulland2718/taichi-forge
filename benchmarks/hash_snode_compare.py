"""Hash SNode correctness and performance comparison.

The script intentionally uses only the Python standard library plus
taichi_forge. Each Taichi case runs in a fresh child process so CPU/CUDA/Vulkan
runtime state and memory counters stay isolated.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


ARCHES = ("cpu", "cuda", "vulkan")
ROOT_LAYOUTS = ("hash", "pointer_bitmasked", "bitmasked", "dense")
TOPOLOGY_LAYOUTS = (
    "hash_dense",
    "pointer_dense",
    "hash_bitmasked",
    "pointer_bitmasked_2d",
    "hash_dynamic",
    "pointer_dynamic",
    "hash_pointer",
    "pointer_pointer",
    "nested_hash",
    "pointer_hash",
    "dynamic_hash",
    "dynamic_bitmasked",
)
LAYOUTS = ROOT_LAYOUTS + TOPOLOGY_LAYOUTS
EXTERNAL_BASELINE_CASES = (
    "warp_hash_cpu",
    "warp_hash_cuda",
    "warp_hash_cpu_2d",
    "warp_hash_cuda_2d",
    "torch_sparse_cpu",
    "torch_sparse_cuda",
    "torch_sparse_cpu_2d",
    "torch_sparse_cuda_2d",
)
RESULT_DIR = ROOT / "benchmarks" / "results" / "hash_snode"
MEMORY_COUNTER_FIELDS = (
    "process_private_mb",
    "gpu_dedicated_mb",
    "gpu_shared_mb",
    "nvidia_smi_compute_mb",
)


def make_key(i: int, domain: int) -> int:
    return (i * 131071 + 17) % domain


def make_value(key: int) -> int:
    return key % 97 + 1


def make_inner_key(i: int, domain: int) -> int:
    return (i * 17 + 3) % domain


def make_value_2d(outer: int, inner: int) -> int:
    return (outer * 31 + inner * 17) % 97 + 1


def expected(active: int, domain: int) -> dict[str, int]:
    keys = [make_key(i, domain) for i in range(active)]
    values = [make_value(k) for k in keys]
    return {
        "count": active,
        "key_sum": sum(keys),
        "value_sum": sum(values),
    }


def expected_2d(
    active: int,
    outer_domain: int,
    inner_active: int,
    inner_domain: int,
    dynamic_inner: bool = False,
) -> dict[str, int]:
    outer_keys = [make_key(i, outer_domain) for i in range(active)]
    if dynamic_inner:
        inner_keys = list(range(inner_active))
    else:
        inner_keys = [make_inner_key(i, inner_domain) for i in range(inner_active)]
    coord_sum = 0
    value_sum = 0
    for outer in outer_keys:
        for inner in inner_keys:
            coord_sum += outer * inner_domain + inner
            value_sum += make_value_2d(outer, inner)
    return {
        "count": active * inner_active,
        "coord_sum": coord_sum,
        "value_sum": value_sum,
    }


def stats_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    mean = statistics.fmean(samples)
    return {
        "samples": len(samples),
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "cv_pct": statistics.pstdev(samples) / mean * 100.0 if mean else 0.0,
    }


def clear_taichi_kernel_profile(ti) -> None:
    try:
        ti.profiler.clear_kernel_profiler_info()
    except Exception:
        pass


def collect_taichi_kernel_profile() -> dict:
    try:
        from taichi_forge.lang import impl

        prog = impl.get_runtime().prog
        prog.sync_kernel_profiler()
        prog.update_kernel_profiler()
        records = prog.get_kernel_profiler_records()
    except Exception as exc:
        return {
            "schema_version": 1,
            "available": False,
            "error": repr(exc),
        }

    by_name: dict[str, dict[str, float | int | str]] = {}
    total_time_ms = 0.0
    for record in records:
        name = str(record.name)
        kernel_time_ms = float(record.kernel_time)
        total_time_ms += kernel_time_ms
        item = by_name.get(name)
        if item is None:
            item = {
                "name": name,
                "count": 0,
                "total_ms": 0.0,
                "min_ms": kernel_time_ms,
                "max_ms": kernel_time_ms,
            }
            by_name[name] = item
        item["count"] = int(item["count"]) + 1
        item["total_ms"] = float(item["total_ms"]) + kernel_time_ms
        item["min_ms"] = min(float(item["min_ms"]), kernel_time_ms)
        item["max_ms"] = max(float(item["max_ms"]), kernel_time_ms)

    kernels = []
    for item in by_name.values():
        count = int(item["count"])
        total = float(item["total_ms"])
        item["avg_ms"] = total / count if count else 0.0
        kernels.append(item)
    kernels.sort(key=lambda item: float(item["total_ms"]), reverse=True)
    return {
        "schema_version": 1,
        "available": True,
        "record_count": len(records),
        "total_time_ms": total_time_ms,
        "kernels": kernels,
    }


def reset_hash_runtime_probe_stats() -> None:
    try:
        from taichi_forge.lang import impl

        impl.get_runtime().prog.reset_hash_snode_probe_stats()
    except Exception:
        pass


def collect_hash_runtime_probe_stats() -> dict:
    try:
        from taichi_forge.lang import impl

        stats = dict(impl.get_runtime().prog.get_hash_snode_probe_stats())
    except Exception as exc:
        return {
            "schema_version": 1,
            "source": "llvm_runtime_global",
            "available": False,
            "error": repr(exc),
        }
    if not stats:
        return {
            "schema_version": 1,
            "source": "llvm_runtime_global",
            "available": False,
            "error": "backend does not expose hash probe stats",
        }
    return {
        "schema_version": 1,
        "source": "llvm_runtime_global",
        "available": True,
        "insert_count": int(stats.get("insert_count", 0)),
        "insert_probe_total": int(stats.get("insert_total", 0)),
        "insert_probe_max": int(stats.get("insert_max", 0)),
        "insert_probe_mean": float(stats.get("insert_mean", 0.0)),
        "lookup_count": int(stats.get("lookup_count", 0)),
        "lookup_probe_total": int(stats.get("lookup_total", 0)),
        "lookup_probe_max": int(stats.get("lookup_max", 0)),
        "lookup_probe_mean": float(stats.get("lookup_mean", 0.0)),
    }


def runtime_probe_max_from_diagnostics(diagnostics: dict) -> int:
    max_probe = -1
    runtime_probe = diagnostics.get("runtime_probe_telemetry", {})
    if not isinstance(runtime_probe, dict):
        return max_probe
    for phase in runtime_probe.values():
        if not isinstance(phase, dict):
            continue
        for key in ("insert_probe_max", "lookup_probe_max"):
            value = phase.get(key)
            if isinstance(value, (int, float)):
                max_probe = max(max_probe, int(value))
    return max_probe


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def estimate_hash_capacity(expected_active: int, load_factor: float = 0.5) -> int:
    return next_power_of_two(max(1, int(math.ceil(expected_active / load_factor))))


def align4(value: int) -> int:
    return (value + 3) & ~3


def hash_mix_u32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def hash_probe_telemetry(keys: list[int], capacity: int) -> dict[str, float | int]:
    if capacity <= 0:
        return {}
    table: list[int | None] = [None] * capacity
    mask = capacity - 1
    insert_probes: list[int] = []
    duplicate_keys = 0
    for key in keys:
        start = hash_mix_u32(key) & mask
        for step in range(capacity):
            bucket = (start + step) & mask
            existing = table[bucket]
            if existing is None:
                table[bucket] = key
                insert_probes.append(step + 1)
                break
            if existing == key:
                duplicate_keys += 1
                insert_probes.append(step + 1)
                break
        else:
            insert_probes.append(capacity)

    occupied = sum(1 for item in table if item is not None)
    collision_count = sum(1 for probes in insert_probes if probes > 1)
    mean_probe = statistics.fmean(insert_probes) if insert_probes else 0.0
    return {
        "schema_version": 1,
        "capacity": capacity,
        "input_keys": len(keys),
        "unique_keys": occupied,
        "duplicate_keys": duplicate_keys,
        "load_factor": occupied / capacity if capacity else 0.0,
        "collision_count": collision_count,
        "collision_rate": collision_count / len(insert_probes) if insert_probes else 0.0,
        "insert_probe_mean": mean_probe,
        "insert_probe_p50": percentile([float(v) for v in insert_probes], 50),
        "insert_probe_p95": percentile([float(v) for v in insert_probes], 95),
        "insert_probe_max": max(insert_probes) if insert_probes else 0,
        "overflow_risk": occupied >= capacity,
    }


def hash_layout_model(
    role: str,
    capacity: int,
    payload_stride: int,
    active_list: bool,
    diagnostics: bool,
    reserved_instances: int = 1,
    active_instances: int | None = None,
    instance_basis: str = "single_container",
) -> dict[str, int | str | bool]:
    include_tombstone = diagnostics or active_list
    payload_stride = align4(max(4, int(payload_stride)))
    state_bytes = capacity * 4
    key_bytes = capacity * 4
    active_count_bytes = 4
    overflow_count_bytes = 4
    active_slots_bytes = capacity * 4 if active_list else 0
    active_slots_count_bytes = 4 if active_list else 0
    tombstone_count_bytes = 4 if include_tombstone else 0
    hash_table_bytes = (
        state_bytes
        + key_bytes
        + active_count_bytes
        + overflow_count_bytes
        + active_slots_bytes
        + active_slots_count_bytes
        + tombstone_count_bytes
    )

    payload_offset = align4(state_bytes + key_bytes)
    active_count_offset = align4(payload_offset + capacity * payload_stride)
    cursor = active_count_offset + active_count_bytes + overflow_count_bytes
    if active_list:
        cursor = align4(cursor) + active_slots_bytes + active_slots_count_bytes
    if include_tombstone:
        cursor = align4(cursor) + tombstone_count_bytes
    ambient_payload_bytes = payload_stride
    container_bytes = align4(cursor) + ambient_payload_bytes

    if active_instances is None:
        active_instances = reserved_instances
    return {
        "role": role,
        "capacity": capacity,
        "payload_stride_bytes": payload_stride,
        "active_list_enabled": active_list,
        "diagnostics_enabled": diagnostics,
        "reserved_instances": max(0, int(reserved_instances)),
        "active_instances": max(0, int(active_instances)),
        "instance_basis": instance_basis,
        "state_bytes": state_bytes,
        "key_bytes": key_bytes,
        "counter_bytes": active_count_bytes
        + overflow_count_bytes
        + active_slots_count_bytes
        + tombstone_count_bytes,
        "active_slots_bytes": active_slots_bytes,
        "hash_table_bytes": hash_table_bytes,
        "hash_payload_reserved_bytes": capacity * payload_stride,
        "ambient_payload_bytes": ambient_payload_bytes,
        "hash_container_bytes": container_bytes,
    }


def reference_layout_model(
    role: str,
    node_type: str,
    capacity: int,
    payload_stride: int,
    reserved_instances: int = 1,
    active_instances: int | None = None,
    instance_basis: str = "single_container",
    include_payload: bool = True,
) -> dict[str, int | str | bool]:
    capacity = max(0, int(capacity))
    payload_stride = align4(max(4, int(payload_stride))) if include_payload else 0
    if active_instances is None:
        active_instances = reserved_instances
    active_instances = max(0, int(active_instances))
    reserved_instances = max(0, int(reserved_instances))

    aux_bytes = 0
    payload_reserved_bytes = capacity * payload_stride
    if node_type == "dense":
        aux_bytes = 0
    elif node_type == "bitmasked":
        aux_bytes = ((capacity + 31) // 32) * 4
    elif node_type == "pointer":
        # LLVM pointer SNodes store a mutex/pointer aux array and a child
        # pointer body array. Child payload is allocated for active slots.
        aux_bytes = capacity * 16
        payload_reserved_bytes = active_instances * payload_stride
    elif node_type == "dynamic":
        # Header pointers plus payload for the logical active part in this
        # benchmark model. Runtime allocator chunks are tracked separately by
        # process/driver counters.
        aux_bytes = 16
        payload_reserved_bytes = active_instances * payload_stride
    else:
        raise RuntimeError(f"unknown reference node type {node_type}")

    return {
        "role": role,
        "node_type": node_type,
        "capacity": capacity,
        "payload_stride_bytes": payload_stride,
        "reserved_instances": reserved_instances,
        "active_instances": active_instances,
        "instance_basis": instance_basis,
        "include_payload": include_payload,
        "aux_bytes": aux_bytes,
        "payload_reserved_bytes": payload_reserved_bytes,
        "container_bytes": aux_bytes + payload_reserved_bytes,
    }


def no_probe_telemetry(structure: str) -> dict[str, float | int | str | bool]:
    return {
        "schema_version": 1,
        "structure": structure,
        "input_keys": 0,
        "unique_keys": 0,
        "load_factor": 0.0,
        "collision_count": 0,
        "collision_rate": 0.0,
        "insert_probe_mean": 0.0,
        "insert_probe_p50": 0.0,
        "insert_probe_p95": 0.0,
        "insert_probe_max": 0,
        "overflow_risk": False,
        "not_applicable": True,
    }


def hash_memory_model(
    hash_nodes: list[dict], reference_nodes: list[dict] | None = None
) -> dict:
    reference_nodes = reference_nodes or []
    table_reserved = 0
    payload_reserved = 0
    container_reserved = 0
    table_active = 0
    for node in hash_nodes:
        reserved_instances = int(node["reserved_instances"])
        active_instances = int(node["active_instances"])
        table_reserved += int(node["hash_table_bytes"]) * reserved_instances
        payload_reserved += int(node["hash_payload_reserved_bytes"]) * reserved_instances
        container_reserved += int(node["hash_container_bytes"]) * reserved_instances
        table_active += int(node["hash_table_bytes"]) * active_instances
    reference_aux_reserved = 0
    reference_payload_reserved = 0
    reference_container_reserved = 0
    for node in reference_nodes:
        reserved_instances = int(node["reserved_instances"])
        reference_aux_reserved += int(node["aux_bytes"]) * reserved_instances
        reference_payload_reserved += (
            int(node["payload_reserved_bytes"]) * reserved_instances
        )
        reference_container_reserved += int(node["container_bytes"]) * reserved_instances
    return {
        "schema_version": 2,
        "source": "benchmark_layout_model",
        "hash_nodes": hash_nodes,
        "reference_nodes": reference_nodes,
        "totals": {
            "hash_table_bytes_reserved_model": table_reserved,
            "hash_payload_bytes_reserved_model": payload_reserved,
            "hash_container_bytes_reserved_model": container_reserved,
            "hash_table_bytes_active_model": table_active,
            "reference_aux_bytes_reserved_model": reference_aux_reserved,
            "reference_payload_bytes_reserved_model": reference_payload_reserved,
            "reference_container_bytes_reserved_model": reference_container_reserved,
            "snode_aux_bytes_reserved_model": table_reserved
            + reference_aux_reserved,
            "snode_payload_bytes_reserved_model": payload_reserved
            + reference_payload_reserved,
            "snode_container_bytes_reserved_model": container_reserved
            + reference_container_reserved,
        },
    }


def hash_case_diagnostics(
    active_count: int,
    expected_active: int,
    domain: int,
    active_list: bool = False,
    diagnostics: bool = False,
    hash_load_factor: float | None = None,
) -> dict[str, float | int | bool | str]:
    load_factor = hash_load_factor or 0.5
    capacity = estimate_hash_capacity(expected_active, load_factor)
    tombstone_count = 0
    overflow_count = 0
    payload_stride = 4  # This benchmark places one i32 scalar directly under hash.
    keys = [make_key(i, domain) for i in range(expected_active)]
    layout = hash_layout_model(
        "root_hash",
        capacity,
        payload_stride,
        active_list,
        diagnostics,
    )
    return {
        "schema_version": 3,
        "capacity": capacity,
        "expected_active": expected_active,
        "observed_active_count": active_count,
        "target_load_factor": load_factor,
        "observed_load_factor": active_count / capacity if capacity else 0.0,
        "payload_stride_bytes": payload_stride,
        "estimated_table_bytes": layout["hash_container_bytes"],
        "estimated_hash_table_bytes": layout["hash_table_bytes"],
        "estimated_hash_payload_reserved_bytes": layout[
            "hash_payload_reserved_bytes"
        ],
        "active_list_enabled": active_list,
        "diagnostics_enabled": diagnostics,
        "tombstone_count": tombstone_count,
        "overflow_count": overflow_count,
        "rebuild_recommended": tombstone_count > 0 or overflow_count > 0,
        "listgen_mode": "active_slots_opt_in" if active_list else "capacity_scan",
        "probe_telemetry": {
            "root_hash": hash_probe_telemetry(keys, capacity),
        },
        "source": "benchmark_hook",
    }


def process_private_mb() -> float:
    if os.name != "nt":
        return -1.0

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi.dll")
    kernel32 = ctypes.WinDLL("kernel32.dll")
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return -1.0
    return counters.PrivateUsage / 1048576.0


def gpu_process_memory_counter_mb(pid: int | None = None, counter: str = "Dedicated Usage") -> float:
    if os.name != "nt":
        return -1.0
    if pid is None:
        pid = os.getpid()
    ps = (
        "$pidToFind = " + str(int(pid)) + "; "
        "$pattern = 'pid_' + $pidToFind + '_*'; "
        "$sum = 0; "
        "try { "
        f"  (Get-Counter '\\GPU Process Memory(*)\\{counter}').CounterSamples | "
        "    Where-Object { $_.InstanceName -like $pattern } | "
        "    ForEach-Object { $sum += $_.CookedValue }; "
        "  [Console]::WriteLine([math]::Round($sum / 1MB, 3)) "
        "} catch { [Console]::WriteLine(-1) }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        return float(out.splitlines()[-1]) if out else -1.0
    except BaseException:
        return -1.0


def nvidia_smi_compute_mb(pid: int | None = None) -> float:
    if pid is None:
        pid = os.getpid()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except BaseException:
        return -1.0
    total = 0.0
    seen = False
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(int(pid)):
            try:
                total += float(parts[1])
            except ValueError:
                continue
            seen = True
    return total if seen else -1.0


def sample_memory() -> dict[str, float]:
    return {
        "process_private_mb": process_private_mb(),
        "gpu_dedicated_mb": gpu_process_memory_counter_mb(counter="Dedicated Usage"),
        "gpu_shared_mb": gpu_process_memory_counter_mb(counter="Shared Usage"),
        "nvidia_smi_compute_mb": nvidia_smi_compute_mb(),
    }


def external_baseline_unavailable(
    case: str, arch: str, workload: str, framework: str, reason: str
) -> dict:
    return {
        "case": case,
        "schema_version": 1,
        "arch": arch,
        "layout": framework,
        "workload": workload,
        "framework": framework,
        "external_baseline": True,
        "available": False,
        "skipped": True,
        "skip_reason": reason,
        "ok": True,
        "memory": sample_memory(),
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
    }


def torch_sync(torch_module, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def torch_sparse_storage_model(indices, values) -> dict[str, int | str]:
    indices_bytes = indices.numel() * indices.element_size()
    values_bytes = values.numel() * values.element_size()
    return {
        "schema_version": 1,
        "source": "torch_sparse_coo_tensor",
        "indices_bytes": int(indices_bytes),
        "values_bytes": int(values_bytes),
        "total_bytes": int(indices_bytes + values_bytes),
    }


def run_warp_hash_case(
    case: str,
    device: str,
    active: int,
    domain: int,
    steps: int,
    warmup: int,
    batch: int,
) -> dict:
    try:
        import hash_snode_warp_baseline as warp_baseline
    except BaseException as exc:
        return external_baseline_unavailable(
            case, device, "root_1d", "warp_open_addressing_hash", repr(exc)
        )

    exp = expected(active, domain)
    try:
        raw = warp_baseline.run_root(
            device, active, domain, steps, warmup, batch, sample_memory
        )
    except BaseException as exc:
        return external_baseline_unavailable(
            case, device, "root_1d", "warp_open_addressing_hash", repr(exc)
        )
    result = {
        "count": int(raw["raw"][0]),
        "key_sum": int(raw["raw"][1]),
        "value_sum": int(raw["raw"][2]),
    }
    first_result = {
        "count": int(raw["first_raw"][0]),
        "key_sum": int(raw["first_raw"][1]),
        "value_sum": int(raw["first_raw"][2]),
    }
    return {
        "case": case,
        "schema_version": 1,
        "arch": device,
        "layout": "warp_open_addressing_hash",
        "workload": "root_1d",
        "framework": "warp",
        "external_baseline": True,
        "available": True,
        "ok": result == exp and first_result == exp,
        "compile_first_s": raw["compile_first_s"],
        "first_result": first_result,
        "result": result,
        "expected": exp,
        "write": stats_ms(raw["write_samples"]),
        "reduce": stats_ms(raw["reduce_samples"]),
        "memory": raw["memory"],
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
        "external_memory_model": raw["external_memory_model"],
    }


def run_warp_hash_case_2d(
    case: str,
    device: str,
    active: int,
    domain: int,
    inner_active: int,
    inner_domain: int,
    steps: int,
    warmup: int,
    batch: int,
) -> dict:
    try:
        import hash_snode_warp_baseline as warp_baseline
    except BaseException as exc:
        return external_baseline_unavailable(
            case, device, "topology_2d", "warp_open_addressing_hash", repr(exc)
        )

    exp = expected_2d(active, domain, inner_active, inner_domain)
    try:
        raw = warp_baseline.run_topology_2d(
            device,
            active,
            domain,
            inner_active,
            inner_domain,
            steps,
            warmup,
            batch,
            sample_memory,
        )
    except BaseException as exc:
        return external_baseline_unavailable(
            case, device, "topology_2d", "warp_open_addressing_hash", repr(exc)
        )
    result = {
        "count": int(raw["raw"][0]),
        "coord_sum": int(raw["raw"][1]),
        "value_sum": int(raw["raw"][2]),
    }
    first_result = {
        "count": int(raw["first_raw"][0]),
        "coord_sum": int(raw["first_raw"][1]),
        "value_sum": int(raw["first_raw"][2]),
    }
    return {
        "case": case,
        "schema_version": 1,
        "arch": device,
        "layout": "warp_open_addressing_hash",
        "workload": "topology_2d",
        "framework": "warp",
        "external_baseline": True,
        "available": True,
        "ok": result == exp and first_result == exp,
        "compile_first_s": raw["compile_first_s"],
        "first_result": first_result,
        "result": result,
        "expected": exp,
        "write": stats_ms(raw["write_samples"]),
        "reduce": stats_ms(raw["reduce_samples"]),
        "memory": raw["memory"],
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
        "external_memory_model": raw["external_memory_model"],
    }


def run_torch_sparse_case(
    case: str,
    device: str,
    active: int,
    domain: int,
    steps: int,
    warmup: int,
    batch: int,
) -> dict:
    try:
        import torch
    except BaseException as exc:
        return external_baseline_unavailable(
            case, device, "root_1d", "torch_sparse_coo", repr(exc)
        )
    if device == "cuda" and not torch.cuda.is_available():
        return external_baseline_unavailable(
            case, device, "root_1d", "torch_sparse_coo", "torch CUDA unavailable"
        )

    exp = expected(active, domain)
    torch_device = torch.device(device)
    key_values = [make_key(i, domain) for i in range(active)]
    value_values = [make_value(k) for k in key_values]

    def build_tensor():
        indices = torch.tensor([key_values], dtype=torch.int64, device=torch_device)
        values = torch.tensor(value_values, dtype=torch.int32, device=torch_device)
        return torch.sparse_coo_tensor(indices, values, (domain,), device=torch_device).coalesce()

    def reduce_tensor(tensor):
        tensor = tensor.coalesce()
        indices = tensor.indices()
        values = tensor.values().to(torch.int64)
        return {
            "count": int(values.numel()),
            "key_sum": int(indices[0].to(torch.int64).sum().item()),
            "value_sum": int(values.sum().item()),
        }

    t0 = time.perf_counter()
    tensor = build_tensor()
    result = reduce_tensor(tensor)
    torch_sync(torch, device)
    compile_s = time.perf_counter() - t0
    ok = result == exp
    mem_after_first = sample_memory()

    for _ in range(warmup):
        tensor = build_tensor()
    torch_sync(torch, device)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            tensor = build_tensor()
        torch_sync(torch, device)
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    for _ in range(warmup):
        reduce_tensor(tensor)
    torch_sync(torch, device)
    reduce_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            result = reduce_tensor(tensor)
        torch_sync(torch, device)
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    return {
        "case": case,
        "schema_version": 1,
        "arch": device,
        "layout": "torch_sparse_coo",
        "workload": "root_1d",
        "framework": "torch",
        "external_baseline": True,
        "available": True,
        "ok": ok and result == exp,
        "compile_first_s": compile_s,
        "result": result,
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": {
            "after_first": mem_after_first,
            "after_bench": sample_memory(),
        },
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
        "external_memory_model": torch_sparse_storage_model(
            tensor.indices(), tensor.values()
        ),
    }


def run_torch_sparse_case_2d(
    case: str,
    device: str,
    active: int,
    domain: int,
    inner_active: int,
    inner_domain: int,
    steps: int,
    warmup: int,
    batch: int,
) -> dict:
    try:
        import torch
    except BaseException as exc:
        return external_baseline_unavailable(
            case, device, "topology_2d", "torch_sparse_coo", repr(exc)
        )
    if device == "cuda" and not torch.cuda.is_available():
        return external_baseline_unavailable(
            case, device, "topology_2d", "torch_sparse_coo", "torch CUDA unavailable"
        )

    exp = expected_2d(active, domain, inner_active, inner_domain)
    torch_device = torch.device(device)
    outer_values = []
    inner_values = []
    value_values = []
    for i in range(active):
        outer = make_key(i, domain)
        for j in range(inner_active):
            inner = make_inner_key(j, inner_domain)
            outer_values.append(outer)
            inner_values.append(inner)
            value_values.append(make_value_2d(outer, inner))

    def build_tensor():
        indices = torch.tensor(
            [outer_values, inner_values], dtype=torch.int64, device=torch_device
        )
        values = torch.tensor(value_values, dtype=torch.int32, device=torch_device)
        return torch.sparse_coo_tensor(
            indices, values, (domain, inner_domain), device=torch_device
        ).coalesce()

    def reduce_tensor(tensor):
        tensor = tensor.coalesce()
        indices = tensor.indices().to(torch.int64)
        values = tensor.values().to(torch.int64)
        coord = indices[0] * inner_domain + indices[1]
        return {
            "count": int(values.numel()),
            "coord_sum": int(coord.sum().item()),
            "value_sum": int(values.sum().item()),
        }

    t0 = time.perf_counter()
    tensor = build_tensor()
    result = reduce_tensor(tensor)
    torch_sync(torch, device)
    compile_s = time.perf_counter() - t0
    ok = result == exp
    mem_after_first = sample_memory()

    for _ in range(warmup):
        tensor = build_tensor()
    torch_sync(torch, device)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            tensor = build_tensor()
        torch_sync(torch, device)
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    for _ in range(warmup):
        reduce_tensor(tensor)
    torch_sync(torch, device)
    reduce_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            result = reduce_tensor(tensor)
        torch_sync(torch, device)
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    return {
        "case": case,
        "schema_version": 1,
        "arch": device,
        "layout": "torch_sparse_coo",
        "workload": "topology_2d",
        "framework": "torch",
        "external_baseline": True,
        "available": True,
        "ok": ok and result == exp,
        "compile_first_s": compile_s,
        "result": result,
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": {
            "after_first": mem_after_first,
            "after_bench": sample_memory(),
        },
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
        "external_memory_model": torch_sparse_storage_model(
            tensor.indices(), tensor.values()
        ),
    }


def run_python_dict(active: int, domain: int, steps: int, warmup: int, batch: int) -> dict:
    exp = expected(active, domain)
    t0 = time.perf_counter()
    data = {}
    for i in range(active):
        key = make_key(i, domain)
        data[key] = make_value(key)
    count = len(data)
    key_sum = sum(data.keys())
    value_sum = sum(data.values())
    compile_s = time.perf_counter() - t0
    ok = {"count": count, "key_sum": key_sum, "value_sum": value_sum} == exp

    for _ in range(warmup):
        for i in range(active):
            key = make_key(i, domain)
            data[key] = make_value(key)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            for i in range(active):
                key = make_key(i, domain)
                data[key] = make_value(key)
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    reduce_samples = []
    for _ in range(warmup):
        count = len(data)
        key_sum = sum(data.keys())
        value_sum = sum(data.values())
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            count = len(data)
            key_sum = sum(data.keys())
            value_sum = sum(data.values())
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    return {
        "case": "python_dict",
        "schema_version": 2,
        "arch": "python",
        "layout": "dict",
        "workload": "root_1d",
        "ok": ok,
        "compile_first_s": compile_s,
        "result": {"count": count, "key_sum": key_sum, "value_sum": value_sum},
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": sample_memory(),
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
    }


def run_python_dict_2d(
    active: int,
    domain: int,
    inner_active: int,
    inner_domain: int,
    steps: int,
    warmup: int,
    batch: int,
) -> dict:
    exp = expected_2d(active, domain, inner_active, inner_domain)

    def write_data(data: dict[tuple[int, int], int]) -> None:
        for i in range(active):
            outer = make_key(i, domain)
            for j in range(inner_active):
                inner = make_inner_key(j, inner_domain)
                data[(outer, inner)] = make_value_2d(outer, inner)

    def reduce_data(data: dict[tuple[int, int], int]) -> dict[str, int]:
        coord_sum = 0
        value_sum = 0
        for (outer, inner), value in data.items():
            coord_sum += outer * inner_domain + inner
            value_sum += value
        return {
            "count": len(data),
            "coord_sum": coord_sum,
            "value_sum": value_sum,
        }

    t0 = time.perf_counter()
    data: dict[tuple[int, int], int] = {}
    write_data(data)
    result = reduce_data(data)
    compile_s = time.perf_counter() - t0
    ok = result == exp

    for _ in range(warmup):
        write_data(data)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            write_data(data)
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    for _ in range(warmup):
        reduce_data(data)
    reduce_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            result = reduce_data(data)
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    return {
        "case": "python_dict_2d",
        "schema_version": 3,
        "arch": "python",
        "layout": "dict_2d",
        "workload": "topology_2d",
        "ok": ok and result == exp,
        "compile_first_s": compile_s,
        "result": result,
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": sample_memory(),
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
    }


def run_taichi_topology_case_initialized(
    ti,
    arch_name: str,
    layout: str,
    active: int,
    domain: int,
    inner_active: int,
    inner_domain: int,
    steps: int,
    warmup: int,
    batch: int,
    mem_after_init: dict[str, float],
    hash_active_list: bool,
    hash_diagnostics: bool,
    outer_hash_load_factor: float | None,
    inner_hash_load_factor: float | None,
    kernel_profiler: bool,
) -> dict:
    if inner_active > inner_domain:
        raise RuntimeError("inner-active must be <= inner-domain")

    dynamic_child = layout in ("hash_dynamic", "pointer_dynamic")
    exp = expected_2d(
        active,
        domain,
        inner_active,
        inner_domain,
        dynamic_inner=dynamic_child,
    )
    if arch_name == "vulkan" and exp["coord_sum"] > 2_000_000_000:
        raise RuntimeError(
            "Vulkan topology benchmark coord_sum exceeds the i32 accumulator "
            "range; use smaller active/domain/inner-domain values."
        )
    acc_dtype = ti.i32 if arch_name == "vulkan" else ti.i64

    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    coord_sum = ti.field(acc_dtype, shape=())
    value_sum = ti.field(acc_dtype, shape=())

    if layout.startswith("hash_") or layout == "nested_hash":
        outer = ti.root.hash(
            ti.i,
            domain,
            expected_active=active,
            hash_load_factor=outer_hash_load_factor,
        )
    elif layout.startswith("pointer_"):
        outer = ti.root.pointer(ti.i, domain)
    elif layout.startswith("dynamic_"):
        outer = ti.root.dynamic(ti.i, domain, chunk_size=min(64, max(1, domain)))
    else:
        raise RuntimeError(f"unknown topology layout {layout}")

    if layout in ("hash_dense", "pointer_dense"):
        leaf = outer.dense(ti.j, inner_domain)
    elif layout in ("hash_bitmasked", "pointer_bitmasked_2d", "dynamic_bitmasked"):
        leaf = outer.bitmasked(ti.j, inner_domain)
    elif layout in ("hash_dynamic", "pointer_dynamic"):
        leaf = outer.dynamic(ti.j, inner_domain, chunk_size=min(8, max(1, inner_domain)))
    elif layout in ("hash_pointer", "pointer_pointer"):
        leaf = outer.pointer(ti.j, inner_domain)
    elif layout in ("nested_hash", "pointer_hash", "dynamic_hash"):
        leaf = outer.hash(
            ti.j,
            inner_domain,
            expected_active=inner_active,
            hash_load_factor=inner_hash_load_factor,
        )
    else:
        raise RuntimeError(f"unknown topology layout {layout}")
    leaf.place(x)

    @ti.func
    def bench_outer_key(i):
        return (i * 131071 + 17) % domain

    @ti.func
    def bench_inner_key(i):
        return (i * 17 + 3) % inner_domain

    @ti.func
    def bench_value_2d(outer_key, inner_key):
        return (outer_key * 31 + inner_key * 17) % 97 + 1

    if dynamic_child:

        @ti.kernel
        def write():
            for p in range(active):
                outer_key = bench_outer_key(p)
                ti.activate(outer, [outer_key])
                ti.deactivate(leaf, [outer_key])
                for q in range(inner_active):
                    ti.append(leaf, outer_key, bench_value_2d(outer_key, q))

    else:

        @ti.kernel
        def write():
            for p in range(active):
                outer_key = bench_outer_key(p)
                for q in range(inner_active):
                    inner_key = bench_inner_key(q)
                    x[outer_key, inner_key] = bench_value_2d(outer_key, inner_key)

    @ti.kernel
    def clear_acc():
        count[None] = 0
        coord_sum[None] = 0
        value_sum[None] = 0

    @ti.kernel
    def reduce():
        for i, j in x:
            value = x[i, j]
            if value != 0:
                count[None] += 1
                coord_sum[None] += i * inner_domain + j
                value_sum[None] += value

    t0 = time.perf_counter()
    write()
    clear_acc()
    reduce()
    ti.sync()
    compile_first_s = time.perf_counter() - t0
    mem_after_first = sample_memory()

    result = {
        "count": int(count[None]),
        "coord_sum": int(coord_sum[None]),
        "value_sum": int(value_sum[None]),
    }
    ok = result == exp

    for _ in range(warmup):
        write()
    ti.sync()
    runtime_probe_telemetry = {}
    if hash_diagnostics and "hash" in layout:
        reset_hash_runtime_probe_stats()
    if kernel_profiler:
        clear_taichi_kernel_profile(ti)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            write()
        ti.sync()
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)
    if hash_diagnostics and "hash" in layout:
        runtime_probe_telemetry["write"] = collect_hash_runtime_probe_stats()
    kernel_profile = {}
    if kernel_profiler:
        kernel_profile["write"] = collect_taichi_kernel_profile()

    for _ in range(warmup):
        clear_acc()
        reduce()
    ti.sync()
    if hash_diagnostics and "hash" in layout:
        reset_hash_runtime_probe_stats()
    if kernel_profiler:
        clear_taichi_kernel_profile(ti)
    reduce_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            clear_acc()
            reduce()
        ti.sync()
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)
    if hash_diagnostics and "hash" in layout:
        runtime_probe_telemetry["reduce"] = collect_hash_runtime_probe_stats()
    if kernel_profiler:
        kernel_profile["reduce"] = collect_taichi_kernel_profile()

    clear_acc()
    reduce()
    ti.sync()
    final_result = {
        "count": int(count[None]),
        "coord_sum": int(coord_sum[None]),
        "value_sum": int(value_sum[None]),
    }

    outer_hash_capacity = estimate_hash_capacity(active, outer_hash_load_factor or 0.5)
    inner_hash_capacity = estimate_hash_capacity(inner_active, inner_hash_load_factor or 0.5)
    hash_nodes = []
    reference_nodes = []
    probe_telemetry = {}
    outer_keys = [make_key(i, domain) for i in range(active)]
    inner_keys = [make_inner_key(i, inner_domain) for i in range(inner_active)]
    outer_active_instances = len(set(outer_keys))
    if layout.startswith("pointer_"):
        reference_nodes.append(
            reference_layout_model(
                "outer_pointer",
                "pointer",
                domain,
                int(outer._cell_size_bytes),
                reserved_instances=1,
                active_instances=outer_active_instances,
                instance_basis="root_container",
                include_payload=layout != "pointer_hash",
            )
        )
    elif layout.startswith("dynamic_"):
        reference_nodes.append(
            reference_layout_model(
                "outer_dynamic",
                "dynamic",
                domain,
                int(outer._cell_size_bytes),
                reserved_instances=1,
                active_instances=outer_active_instances,
                instance_basis="root_container",
                include_payload=layout != "dynamic_hash",
            )
        )
    if layout.startswith("hash_") or layout == "nested_hash":
        outer_payload_stride = int(outer._cell_size_bytes)
        hash_nodes.append(
            hash_layout_model(
                "outer_hash",
                outer_hash_capacity,
                outer_payload_stride,
                hash_active_list,
                hash_diagnostics,
                reserved_instances=1,
                active_instances=1,
                instance_basis="root_container",
            )
        )
        probe_telemetry["outer_hash"] = hash_probe_telemetry(
            outer_keys, outer_hash_capacity
        )
    if layout in ("nested_hash", "pointer_hash", "dynamic_hash"):
        inner_payload_stride = int(leaf._cell_size_bytes)
        if layout == "nested_hash":
            reserved_instances = outer_hash_capacity
            instance_basis = "outer_hash_capacity"
        else:
            reserved_instances = outer_active_instances
            instance_basis = "active_outer_parent_instances"
        hash_nodes.append(
            hash_layout_model(
                "inner_hash",
                inner_hash_capacity,
                inner_payload_stride,
                hash_active_list,
                hash_diagnostics,
                reserved_instances=reserved_instances,
                active_instances=outer_active_instances,
                instance_basis=instance_basis,
            )
        )
        probe_telemetry["inner_hash"] = hash_probe_telemetry(
            inner_keys, inner_hash_capacity
        )

    result_payload = {
        "case": f"{arch_name}_{layout}",
        "schema_version": 3,
        "arch": arch_name,
        "layout": layout,
        "workload": "topology_2d",
        "ok": ok and final_result == exp,
        "compile_first_s": compile_first_s,
        "first_result": result,
        "first_ok": ok,
        "result": final_result,
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": {
            "after_init": mem_after_init,
            "after_first": mem_after_first,
            "after_bench": sample_memory(),
        },
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
        "benchmark_config": {
            "active": active,
            "domain": domain,
            "inner_active": inner_active,
            "inner_domain": inner_domain,
            "kernel_profiler": kernel_profiler,
            "outer_hash_load_factor": outer_hash_load_factor,
            "inner_hash_load_factor": inner_hash_load_factor,
        },
    }
    if kernel_profile:
        result_payload["kernel_profile"] = kernel_profile
    if hash_nodes or reference_nodes:
        result_payload["snode_memory_model"] = hash_memory_model(
            hash_nodes, reference_nodes
        )
    if "hash" in layout:
        result_payload["hash_diagnostics"] = {
            "schema_version": 4,
            "source": "topology_benchmark",
            "outer_expected_active": active,
            "outer_domain": domain,
            "inner_expected_active": inner_active,
            "inner_domain": inner_domain,
            "layout": layout,
            "dynamic_child": dynamic_child,
            "probe_telemetry": probe_telemetry,
        }
        if runtime_probe_telemetry:
            result_payload["hash_diagnostics"][
                "runtime_probe_telemetry"
            ] = runtime_probe_telemetry
    else:
        result_payload["reference_diagnostics"] = {
            "schema_version": 1,
            "source": "topology_benchmark",
            "layout": layout,
            "probe_telemetry": {
                "reference": no_probe_telemetry("non_hash_reference"),
            },
        }
    return result_payload


def run_taichi_case(
    arch_name: str,
    layout: str,
    active: int,
    domain: int,
    inner_active: int,
    inner_domain: int,
    steps: int,
    warmup: int,
    batch: int,
    hash_active_list: bool,
    hash_diagnostics: bool,
    outer_hash_load_factor: float | None,
    inner_hash_load_factor: float | None,
    kernel_profiler: bool,
) -> dict:
    import taichi_forge as ti

    arch = getattr(ti, arch_name)
    init_kwargs = {
        "arch": arch,
        "offline_cache": False,
        "debug": False,
        "log_level": "warn",
        "hash_snode_experimental": True,
        "hash_snode_active_list": hash_active_list,
        "hash_snode_diagnostics": hash_diagnostics,
        "kernel_profiler": kernel_profiler,
    }
    if arch_name == "vulkan":
        init_kwargs.update(
            {
                "vulkan_sparse_experimental": True,
                "vulkan_listgen_dynamic_size": True,
            }
        )
    ti.init(**init_kwargs)

    mem_after_init = sample_memory()

    max_key_sum = active * max(0, domain - 1)
    if arch_name == "vulkan" and max_key_sum > 2_000_000_000:
        raise RuntimeError(
            "Vulkan hash benchmark key_sum exceeds the i32 accumulator range; "
            "use a smaller active/domain pair for Vulkan until i64 scalar "
            "atomics are part of the benchmark target."
        )
    acc_dtype = ti.i32 if arch_name == "vulkan" else ti.i64

    if layout in TOPOLOGY_LAYOUTS:
        return run_taichi_topology_case_initialized(
            ti,
            arch_name,
            layout,
            active,
            domain,
            inner_active,
            inner_domain,
            steps,
            warmup,
            batch,
            mem_after_init,
            hash_active_list,
            hash_diagnostics,
            outer_hash_load_factor,
            inner_hash_load_factor,
            kernel_profiler,
        )

    x = ti.field(ti.i32)
    count = ti.field(ti.i32, shape=())
    key_sum = ti.field(acc_dtype, shape=())
    value_sum = ti.field(acc_dtype, shape=())

    hash_node = None
    reference_root = None
    reference_leaf = None
    if layout == "hash":
        hash_node = ti.root.hash(
            ti.i,
            domain,
            expected_active=active,
            hash_load_factor=outer_hash_load_factor,
        )
        hash_node.place(x)
    elif layout == "pointer_bitmasked":
        block = 64
        reference_root = ti.root.pointer(ti.i, max(1, domain // block))
        reference_leaf = reference_root.bitmasked(ti.i, block)
        reference_leaf.place(x)
    elif layout == "bitmasked":
        reference_root = ti.root.bitmasked(ti.i, domain)
        reference_root.place(x)
    elif layout == "dense":
        reference_root = ti.root.dense(ti.i, domain)
        reference_root.place(x)
    else:
        raise RuntimeError(f"unknown layout {layout}")

    @ti.func
    def bench_key(i):
        return (i * 131071 + 17) % domain

    @ti.func
    def bench_value(k):
        return k % 97 + 1

    @ti.kernel
    def write():
        for p in range(active):
            key = bench_key(p)
            x[key] = bench_value(key)

    @ti.kernel
    def clear_acc():
        count[None] = 0
        key_sum[None] = 0
        value_sum[None] = 0

    if layout == "dense":

        @ti.kernel
        def reduce():
            for i in x:
                v = x[i]
                if v != 0:
                    count[None] += 1
                    key_sum[None] += i
                    value_sum[None] += v

    else:

        @ti.kernel
        def reduce():
            for i in x:
                count[None] += 1
                key_sum[None] += i
                value_sum[None] += x[i]

    exp = expected(active, domain)

    t0 = time.perf_counter()
    write()
    clear_acc()
    reduce()
    ti.sync()
    compile_first_s = time.perf_counter() - t0
    mem_after_first = sample_memory()

    result = {
        "count": int(count[None]),
        "key_sum": int(key_sum[None]),
        "value_sum": int(value_sum[None]),
    }
    ok = result == exp

    for _ in range(warmup):
        write()
    ti.sync()
    runtime_probe_telemetry = {}
    if hash_diagnostics and layout == "hash":
        reset_hash_runtime_probe_stats()
    if kernel_profiler:
        clear_taichi_kernel_profile(ti)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            write()
        ti.sync()
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)
    if hash_diagnostics and layout == "hash":
        runtime_probe_telemetry["write"] = collect_hash_runtime_probe_stats()
    kernel_profile = {}
    if kernel_profiler:
        kernel_profile["write"] = collect_taichi_kernel_profile()

    for _ in range(warmup):
        clear_acc()
        reduce()
    ti.sync()
    if hash_diagnostics and layout == "hash":
        reset_hash_runtime_probe_stats()
    if kernel_profiler:
        clear_taichi_kernel_profile(ti)
    reduce_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            clear_acc()
            reduce()
        ti.sync()
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)
    if hash_diagnostics and layout == "hash":
        runtime_probe_telemetry["reduce"] = collect_hash_runtime_probe_stats()
    if kernel_profiler:
        kernel_profile["reduce"] = collect_taichi_kernel_profile()

    clear_acc()
    reduce()
    ti.sync()
    final_result = {
        "count": int(count[None]),
        "key_sum": int(key_sum[None]),
        "value_sum": int(value_sum[None]),
    }

    hash_nodes = []
    reference_nodes = []
    probe_telemetry = {}
    if hash_node is not None:
        capacity = estimate_hash_capacity(active, outer_hash_load_factor or 0.5)
        payload_stride = int(hash_node._cell_size_bytes)
        hash_nodes.append(
            hash_layout_model(
                "root_hash",
                capacity,
                payload_stride,
                hash_active_list,
                hash_diagnostics,
                reserved_instances=1,
                active_instances=1,
                instance_basis="root_container",
            )
        )
        probe_telemetry["root_hash"] = hash_probe_telemetry(
            [make_key(i, domain) for i in range(active)], capacity
        )
    elif layout == "pointer_bitmasked" and reference_root is not None:
        block = 64
        active_blocks = len({make_key(i, domain) // block for i in range(active)})
        reference_nodes.append(
            reference_layout_model(
                "root_pointer",
                "pointer",
                max(1, domain // block),
                int(reference_root._cell_size_bytes),
                reserved_instances=1,
                active_instances=active_blocks,
                instance_basis="root_container",
            )
        )
    elif layout in ("bitmasked", "dense") and reference_root is not None:
        reference_nodes.append(
            reference_layout_model(
                f"root_{layout}",
                layout,
                domain,
                int(reference_root._cell_size_bytes),
                reserved_instances=1,
                active_instances=1,
                instance_basis="root_container",
            )
        )

    result_payload = {
        "case": f"{arch_name}_{layout}",
        "schema_version": 2,
        "arch": arch_name,
        "layout": layout,
        "workload": "root_1d",
        "ok": ok and final_result == exp,
        "compile_first_s": compile_first_s,
        "first_result": result,
        "first_ok": ok,
        "result": final_result,
        "expected": exp,
        "write": stats_ms(write_samples),
        "reduce": stats_ms(reduce_samples),
        "memory": {
            "after_init": mem_after_init,
            "after_first": mem_after_first,
            "after_bench": sample_memory(),
        },
        "memory_counter_fields": list(MEMORY_COUNTER_FIELDS),
        "benchmark_config": {
            "active": active,
            "domain": domain,
            "inner_active": inner_active,
            "inner_domain": inner_domain,
            "kernel_profiler": kernel_profiler,
            "outer_hash_load_factor": outer_hash_load_factor,
            "inner_hash_load_factor": inner_hash_load_factor,
        },
    }
    if kernel_profile:
        result_payload["kernel_profile"] = kernel_profile
    if layout == "hash":
        result_payload["hash_diagnostics"] = hash_case_diagnostics(
            final_result["count"],
            active,
            domain,
            hash_active_list,
            hash_diagnostics,
            outer_hash_load_factor,
        )
        result_payload["hash_diagnostics"]["probe_telemetry"] = probe_telemetry
        if runtime_probe_telemetry:
            result_payload["hash_diagnostics"][
                "runtime_probe_telemetry"
            ] = runtime_probe_telemetry
    else:
        result_payload["reference_diagnostics"] = {
            "schema_version": 1,
            "source": "root_benchmark",
            "layout": layout,
            "probe_telemetry": {
                "reference": no_probe_telemetry("non_hash_reference"),
            },
        }
    if hash_nodes or reference_nodes:
        result_payload["snode_memory_model"] = hash_memory_model(
            hash_nodes, reference_nodes
        )
    return result_payload


def print_table(results: list[dict]) -> None:
    print(
        "\ncase,ok,compile_s,write_median_ms,reduce_median_ms,"
        "proc_mb,gpu_ded_mb,hash_table_kb,snode_container_kb,probe_max,"
        "runtime_probe_max"
    )
    for item in results:
        mem = item.get("memory", {})
        if "after_bench" in mem:
            proc = mem["after_bench"].get("process_private_mb", -1.0)
            gpu = mem["after_bench"].get("gpu_dedicated_mb", -1.0)
        else:
            proc = mem.get("process_private_mb", -1.0)
            gpu = mem.get("gpu_dedicated_mb", -1.0)
        totals = item.get("snode_memory_model", {}).get("totals", {})
        hash_table_kb = (
            float(totals.get("hash_table_bytes_reserved_model", -1024.0)) / 1024.0
        )
        snode_container_kb = (
            float(totals.get("snode_container_bytes_reserved_model", -1024.0))
            / 1024.0
        )
        probe_max = -1
        for diagnostics_key in ("hash_diagnostics", "reference_diagnostics"):
            diagnostics = item.get(diagnostics_key, {})
            for value in diagnostics.get("probe_telemetry", {}).values():
                if isinstance(value, dict):
                    probe_max = max(
                        probe_max, int(value.get("insert_probe_max", -1))
                    )
        runtime_probe_max = runtime_probe_max_from_diagnostics(
            item.get("hash_diagnostics", {})
        )
        print(
            "{case},{ok},{compile:.6f},{write:.6f},{reduce:.6f},"
            "{proc:.3f},{gpu:.3f},{hash_table_kb:.3f},"
            "{snode_container_kb:.3f},{probe_max},{runtime_probe_max}".format(
                case=item.get("case"),
                ok=item.get("ok"),
                compile=item.get("compile_first_s", -1.0),
                write=item.get("write", {}).get("median_ms", -1.0),
                reduce=item.get("reduce", {}).get("median_ms", -1.0),
                proc=proc,
                gpu=gpu,
                hash_table_kb=hash_table_kb,
                snode_container_kb=snode_container_kb,
                probe_max=probe_max,
                runtime_probe_max=runtime_probe_max,
            )
        )


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def relative_range_pct(values: list[float]) -> float:
    if not values:
        return -1.0
    med = statistics.median(values)
    if med == 0:
        return 0.0
    return (max(values) - min(values)) / med * 100.0


def ratio_or_none(value: float | None, baseline: float | None) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    if not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    return float(value) / float(baseline)


def internal_reference_case_for_summary(item: dict) -> str | None:
    arch = item.get("arch")
    workload = item.get("workload")
    if not isinstance(arch, str) or arch == "python":
        return None
    if workload == "topology_2d":
        return f"{arch}_pointer_bitmasked_2d"
    if workload == "root_1d":
        return f"{arch}_pointer_bitmasked"
    return None


def external_baseline_case_for_summary(item: dict) -> str | None:
    arch = item.get("arch")
    layout = item.get("layout")
    workload = item.get("workload")
    if not isinstance(arch, str):
        return None
    if workload == "topology_2d":
        if arch == "cpu":
            return "warp_hash_cpu_2d"
        if arch == "cuda":
            return "warp_hash_cuda_2d"
        if arch == "vulkan" and isinstance(layout, str):
            return f"cuda_{layout}"
        return f"external_{arch}_hash_like_unavailable_2d"
    if workload == "root_1d":
        if arch == "cpu":
            return "warp_hash_cpu"
        if arch == "cuda":
            return "warp_hash_cuda"
        if arch == "vulkan" and isinstance(layout, str):
            return f"cuda_{layout}"
        return f"external_{arch}_hash_like_unavailable"
    return None


def attach_baseline_ratios(summary: list[dict]) -> None:
    by_case = {str(item.get("case")): item for item in summary}
    for item in summary:
        internal_case = None
        if not item.get("external_baseline"):
            internal_case = internal_reference_case_for_summary(item)
        if internal_case is not None:
            internal = by_case.get(internal_case)
            item["internal_reference_case"] = internal_case
            item["internal_reference_missing"] = internal is None
            if internal is not None:
                item["write_vs_internal_reference"] = ratio_or_none(
                    item.get("write_median_ms"), internal.get("write_median_ms")
                )
                item["reduce_vs_internal_reference"] = ratio_or_none(
                    item.get("reduce_median_ms"), internal.get("reduce_median_ms")
                )
                item["compile_first_vs_internal_reference"] = ratio_or_none(
                    item.get("compile_first_median_s"),
                    internal.get("compile_first_median_s"),
                )
                item["snode_container_vs_internal_reference"] = ratio_or_none(
                    item.get("snode_container_reserved_model_bytes"),
                    internal.get("snode_container_reserved_model_bytes"),
                )

        external_case = external_baseline_case_for_summary(item)
        if external_case is None or item.get("external_baseline"):
            continue
        external = by_case.get(external_case)
        item["baseline_case"] = external_case
        item["baseline_kind"] = (
            "cuda_same_layout_reference"
            if item.get("arch") == "vulkan" and external_case.startswith("cuda_")
            else "external_dsl_hash_like"
        )
        item["baseline_missing"] = external is None or not external.get("available", True)
        if external is not None and not item["baseline_missing"]:
            item["baseline_framework"] = external.get("framework")
            item["baseline_layout"] = external.get("layout")
            item["baseline_storage_bytes"] = external.get("external_storage_bytes")
            item["baseline_snode_container_reserved_model_bytes"] = external.get(
                "snode_container_reserved_model_bytes"
            )
            item["write_vs_baseline"] = ratio_or_none(
                item.get("write_median_ms"), external.get("write_median_ms")
            )
            item["reduce_vs_baseline"] = ratio_or_none(
                item.get("reduce_median_ms"), external.get("reduce_median_ms")
            )
            item["compile_first_vs_baseline"] = ratio_or_none(
                item.get("compile_first_median_s"),
                external.get("compile_first_median_s"),
            )
            item["memory_model_vs_external_storage"] = ratio_or_none(
                item.get("snode_container_reserved_model_bytes"),
                external.get("external_storage_bytes"),
            )
            item["memory_model_vs_baseline"] = ratio_or_none(
                item.get("snode_container_reserved_model_bytes"),
                external.get("external_storage_bytes")
                if isinstance(external.get("external_storage_bytes"), (int, float))
                else external.get("snode_container_reserved_model_bytes"),
            )


def summarize_results(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        grouped.setdefault(str(item.get("case")), []).append(item)

    summary = []
    for case, items in sorted(grouped.items()):
        compile_values = [
            float(item["compile_first_s"])
            for item in items
            if isinstance(item.get("compile_first_s"), (int, float))
        ]
        write_values = [
            float(item.get("write", {}).get("median_ms"))
            for item in items
            if isinstance(item.get("write", {}).get("median_ms"), (int, float))
        ]
        reduce_values = [
            float(item.get("reduce", {}).get("median_ms"))
            for item in items
            if isinstance(item.get("reduce", {}).get("median_ms"), (int, float))
        ]
        process_values = []
        gpu_values = []
        nvidia_values = []
        hash_table_reserved_values = []
        hash_payload_reserved_values = []
        hash_container_reserved_values = []
        reference_aux_reserved_values = []
        reference_payload_reserved_values = []
        reference_container_reserved_values = []
        snode_aux_reserved_values = []
        snode_payload_reserved_values = []
        snode_container_reserved_values = []
        external_storage_values = []
        probe_max_values = []
        runtime_probe_max_values = []
        runtime_insert_mean_values = []
        runtime_lookup_mean_values = []
        for item in items:
            memory = item.get("memory", {})
            if "after_bench" in memory:
                memory = memory["after_bench"]
            if isinstance(memory.get("process_private_mb"), (int, float)):
                process_values.append(float(memory["process_private_mb"]))
            if isinstance(memory.get("gpu_dedicated_mb"), (int, float)):
                gpu_values.append(float(memory["gpu_dedicated_mb"]))
            if isinstance(memory.get("nvidia_smi_compute_mb"), (int, float)):
                nvidia_values.append(float(memory["nvidia_smi_compute_mb"]))
            snode_memory = item.get("snode_memory_model", {})
            totals = snode_memory.get("totals", {})
            if isinstance(
                totals.get("hash_table_bytes_reserved_model"), (int, float)
            ):
                hash_table_reserved_values.append(
                    float(totals["hash_table_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("hash_payload_bytes_reserved_model"), (int, float)
            ):
                hash_payload_reserved_values.append(
                    float(totals["hash_payload_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("hash_container_bytes_reserved_model"), (int, float)
            ):
                hash_container_reserved_values.append(
                    float(totals["hash_container_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("reference_aux_bytes_reserved_model"), (int, float)
            ):
                reference_aux_reserved_values.append(
                    float(totals["reference_aux_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("reference_payload_bytes_reserved_model"), (int, float)
            ):
                reference_payload_reserved_values.append(
                    float(totals["reference_payload_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("reference_container_bytes_reserved_model"), (int, float)
            ):
                reference_container_reserved_values.append(
                    float(totals["reference_container_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("snode_aux_bytes_reserved_model"), (int, float)
            ):
                snode_aux_reserved_values.append(
                    float(totals["snode_aux_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("snode_payload_bytes_reserved_model"), (int, float)
            ):
                snode_payload_reserved_values.append(
                    float(totals["snode_payload_bytes_reserved_model"])
                )
            if isinstance(
                totals.get("snode_container_bytes_reserved_model"), (int, float)
            ):
                snode_container_reserved_values.append(
                    float(totals["snode_container_bytes_reserved_model"])
                )
            external_memory = item.get("external_memory_model", {})
            if isinstance(external_memory.get("total_bytes"), (int, float)):
                external_storage_values.append(float(external_memory["total_bytes"]))
            for diagnostics_key in ("hash_diagnostics", "reference_diagnostics"):
                diagnostics = item.get(diagnostics_key, {})
                probe = diagnostics.get("probe_telemetry", {})
                if isinstance(probe, dict):
                    for value in probe.values():
                        if isinstance(value, dict) and isinstance(
                            value.get("insert_probe_max"), (int, float)
                        ):
                            probe_max_values.append(
                                float(value["insert_probe_max"])
                            )
                runtime_probe = diagnostics.get("runtime_probe_telemetry", {})
                if isinstance(runtime_probe, dict):
                    for value in runtime_probe.values():
                        if not isinstance(value, dict):
                            continue
                        for max_key in ("insert_probe_max", "lookup_probe_max"):
                            if isinstance(value.get(max_key), (int, float)):
                                runtime_probe_max_values.append(
                                    float(value[max_key])
                                )
                        if isinstance(value.get("insert_probe_mean"), (int, float)):
                            runtime_insert_mean_values.append(
                                float(value["insert_probe_mean"])
                            )
                        if isinstance(value.get("lookup_probe_mean"), (int, float)):
                            runtime_lookup_mean_values.append(
                                float(value["lookup_probe_mean"])
                            )
        summary.append(
            {
                "case": case,
                "arch": items[0].get("arch"),
                "layout": items[0].get("layout"),
                "workload": items[0].get("workload"),
                "framework": items[0].get("framework"),
                "external_baseline": bool(items[0].get("external_baseline")),
                "available": bool(items[0].get("available", True)),
                "skipped": bool(items[0].get("skipped", False)),
                "skip_reason": items[0].get("skip_reason"),
                "runs": len(items),
                "ok": all(bool(item.get("ok")) for item in items),
                "compile_first_median_s": median_or_none(compile_values),
                "compile_first_range_pct": relative_range_pct(compile_values),
                "write_median_ms": median_or_none(write_values),
                "write_range_pct": relative_range_pct(write_values),
                "reduce_median_ms": median_or_none(reduce_values),
                "reduce_range_pct": relative_range_pct(reduce_values),
                "process_private_max_mb": max(process_values) if process_values else None,
                "gpu_dedicated_max_mb": max(gpu_values) if gpu_values else None,
                "nvidia_smi_compute_max_mb": max(nvidia_values)
                if nvidia_values
                else None,
                "hash_table_reserved_model_bytes": median_or_none(
                    hash_table_reserved_values
                ),
                "hash_payload_reserved_model_bytes": median_or_none(
                    hash_payload_reserved_values
                ),
                "hash_container_reserved_model_bytes": median_or_none(
                    hash_container_reserved_values
                ),
                "reference_aux_reserved_model_bytes": median_or_none(
                    reference_aux_reserved_values
                ),
                "reference_payload_reserved_model_bytes": median_or_none(
                    reference_payload_reserved_values
                ),
                "reference_container_reserved_model_bytes": median_or_none(
                    reference_container_reserved_values
                ),
                "snode_aux_reserved_model_bytes": median_or_none(
                    snode_aux_reserved_values
                ),
                "snode_payload_reserved_model_bytes": median_or_none(
                    snode_payload_reserved_values
                ),
                "snode_container_reserved_model_bytes": median_or_none(
                    snode_container_reserved_values
                ),
                "external_storage_bytes": median_or_none(external_storage_values),
                "probe_insert_max": max(probe_max_values)
                if probe_max_values
                else None,
                "runtime_probe_max": max(runtime_probe_max_values)
                if runtime_probe_max_values
                else None,
                "runtime_insert_probe_mean_max": max(runtime_insert_mean_values)
                if runtime_insert_mean_values
                else None,
                "runtime_lookup_probe_mean_max": max(runtime_lookup_mean_values)
                if runtime_lookup_mean_values
                else None,
            }
        )
    attach_baseline_ratios(summary)
    return summary


def cases_for_suite(suite: str, include_external_baselines: bool = False) -> list[str]:
    cases: list[str] = []
    if suite in ("root", "all"):
        cases.append("python_dict")
        if include_external_baselines:
            cases.extend(
                (
                    "warp_hash_cpu",
                    "warp_hash_cuda",
                    "torch_sparse_cpu",
                    "torch_sparse_cuda",
                )
            )
        cases.extend(f"{arch}:{layout}" for arch in ARCHES for layout in ROOT_LAYOUTS)
    if suite in ("topology", "all"):
        cases.append("python_dict_2d")
        if include_external_baselines:
            cases.extend(
                (
                    "warp_hash_cpu_2d",
                    "warp_hash_cuda_2d",
                    "torch_sparse_cpu_2d",
                    "torch_sparse_cuda_2d",
                )
            )
        cases.extend(f"{arch}:{layout}" for arch in ARCHES for layout in TOPOLOGY_LAYOUTS)
    return cases


def child_main(args: argparse.Namespace) -> int:
    try:
        if args.case == "python_dict":
            result = run_python_dict(args.active, args.domain, args.steps, args.warmup, args.batch)
        elif args.case == "python_dict_2d":
            result = run_python_dict_2d(
                args.active,
                args.domain,
                args.inner_active,
                args.inner_domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "warp_hash_cpu":
            result = run_warp_hash_case(
                args.case,
                "cpu",
                args.active,
                args.domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "warp_hash_cuda":
            result = run_warp_hash_case(
                args.case,
                "cuda",
                args.active,
                args.domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "warp_hash_cpu_2d":
            result = run_warp_hash_case_2d(
                args.case,
                "cpu",
                args.active,
                args.domain,
                args.inner_active,
                args.inner_domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "warp_hash_cuda_2d":
            result = run_warp_hash_case_2d(
                args.case,
                "cuda",
                args.active,
                args.domain,
                args.inner_active,
                args.inner_domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "torch_sparse_cpu":
            result = run_torch_sparse_case(
                args.case,
                "cpu",
                args.active,
                args.domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "torch_sparse_cuda":
            result = run_torch_sparse_case(
                args.case,
                "cuda",
                args.active,
                args.domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "torch_sparse_cpu_2d":
            result = run_torch_sparse_case_2d(
                args.case,
                "cpu",
                args.active,
                args.domain,
                args.inner_active,
                args.inner_domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        elif args.case == "torch_sparse_cuda_2d":
            result = run_torch_sparse_case_2d(
                args.case,
                "cuda",
                args.active,
                args.domain,
                args.inner_active,
                args.inner_domain,
                args.steps,
                args.warmup,
                args.batch,
            )
        else:
            arch, layout = args.case.split(":", 1)
            result = run_taichi_case(
                arch,
                layout,
                args.active,
                args.domain,
                args.inner_active,
                args.inner_domain,
                args.steps,
                args.warmup,
                args.batch,
                args.hash_active_list,
                args.hash_diagnostics,
                args.outer_hash_load_factor,
                args.inner_hash_load_factor,
                args.kernel_profiler,
            )
        print("HASH_SNODE_BENCH_RESULT " + json.dumps(result, sort_keys=True))
        return 0 if result.get("ok") else 2
    except BaseException as exc:
        result = {"case": args.case, "ok": False, "error": repr(exc)}
        print("HASH_SNODE_BENCH_RESULT " + json.dumps(result, sort_keys=True))
        return 1


def parent_main(args: argparse.Namespace) -> int:
    cases = args.cases
    if not cases:
        cases = cases_for_suite(args.suite, args.include_external_baselines)

    results = []
    for repeat_index in range(args.repeat):
        for case in cases:
            child_python = (
                args.external_python
                if args.external_python and case in EXTERNAL_BASELINE_CASES
                else sys.executable
            )
            cmd = [
                child_python,
                str(Path(__file__).resolve()),
                "--child",
                "--case",
                case,
                "--active",
                str(args.active),
                "--domain",
                str(args.domain),
                "--inner-active",
                str(args.inner_active),
                "--inner-domain",
                str(args.inner_domain),
                "--steps",
                str(args.steps),
                "--warmup",
                str(args.warmup),
                "--batch",
                str(args.batch),
            ]
            if args.hash_active_list:
                cmd.append("--hash-active-list")
            if args.hash_diagnostics:
                cmd.append("--hash-diagnostics")
            if args.kernel_profiler:
                cmd.append("--kernel-profiler")
            if args.outer_hash_load_factor is not None:
                cmd.extend(
                    ["--outer-hash-load-factor", str(args.outer_hash_load_factor)]
                )
            if args.inner_hash_load_factor is not None:
                cmd.extend(
                    ["--inner-hash-load-factor", str(args.inner_hash_load_factor)]
                )
            print(
                f"[hash-bench] running {case} repeat={repeat_index}",
                file=sys.stderr,
                flush=True,
            )
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            parsed = None
            for line in proc.stdout.splitlines():
                if line.startswith("HASH_SNODE_BENCH_RESULT "):
                    parsed = json.loads(line.split(" ", 1)[1])
            if parsed is None:
                parsed = {
                    "case": case,
                    "ok": False,
                    "error": "missing result line",
                    "returncode": proc.returncode,
                    "output_tail": proc.stdout.splitlines()[-20:],
                }
            parsed["returncode"] = proc.returncode
            parsed["repeat_index"] = repeat_index
            results.append(parsed)
            print(json.dumps(parsed, sort_keys=True), flush=True)

    output_path = resolve_output_path(args.output)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary_path = output_path.with_name(output_path.stem + "_summary.json")
        summary_path.write_text(
            json.dumps(summarize_results(results), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[hash-bench] wrote {output_path}", file=sys.stderr, flush=True)
        print(f"[hash-bench] wrote {summary_path}", file=sys.stderr, flush=True)
    print_table(results)
    return 0 if all(r.get("ok") for r in results if "error" not in r) else 1


def resolve_output_path(output: str) -> Path | None:
    if not output:
        return None
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return path
    if len(rel.parts) >= 1 and rel.parts[0] == "opt_doc":
        return RESULT_DIR / path.name
    if path.parent == ROOT:
        return RESULT_DIR / path.name
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case", default="")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--suite", choices=("root", "topology", "all"), default="root")
    parser.add_argument("--active", type=int, default=4096)
    parser.add_argument("--domain", type=int, default=65536)
    parser.add_argument("--inner-active", type=int, default=4)
    parser.add_argument("--inner-domain", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--include-external-baselines", action="store_true")
    parser.add_argument(
        "--external-python",
        default="",
        help=(
            "Optional Python executable for external DSL baselines such as "
            "torch_sparse_*; Taichi cases still use the current interpreter."
        ),
    )
    parser.add_argument("--hash-active-list", action="store_true")
    parser.add_argument("--hash-diagnostics", action="store_true")
    parser.add_argument("--kernel-profiler", action="store_true")
    parser.add_argument("--outer-hash-load-factor", type=float, default=None)
    parser.add_argument("--inner-hash-load-factor", type=float, default=None)
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output JSON path. Bare filenames and opt_doc/*.json are written "
            "under benchmarks/results/hash_snode/."
        ),
    )
    args = parser.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
