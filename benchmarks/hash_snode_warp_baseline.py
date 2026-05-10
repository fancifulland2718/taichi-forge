"""Warp hash-like baselines for hash_snode_compare.py.

This module is imported only by external baseline child processes. Keeping the
Warp kernels in a real source file is required because Warp inspects Python
source when compiling kernels.
"""

from __future__ import annotations

import time

import warp as wp


@wp.kernel
def fill_root_kernel(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
    active: int,
    domain: int,
):
    tid = wp.tid()
    if tid >= active:
        return
    key = (tid * 131071 + 17) % domain
    keys[tid] = key
    values[tid] = key % 97 + 1


@wp.kernel
def fill_topology_2d_kernel(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
    entries: int,
    active: int,
    domain: int,
    inner_active: int,
    inner_domain: int,
):
    tid = wp.tid()
    if tid >= entries:
        return
    outer_index = tid // inner_active
    inner_index = tid - outer_index * inner_active
    outer = (outer_index * 131071 + 17) % domain
    inner = (inner_index * 17 + 3) % inner_domain
    keys[tid] = outer * inner_domain + inner
    values[tid] = (outer * 31 + inner * 17) % 97 + 1


@wp.kernel
def hash_insert_kernel(
    input_keys: wp.array(dtype=wp.int32),
    input_values: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.int32),
    table_keys: wp.array(dtype=wp.int32),
    table_values: wp.array(dtype=wp.int32),
    entries: int,
    capacity: int,
):
    tid = wp.tid()
    if tid >= entries:
        return
    key = input_keys[tid]
    value = input_values[tid]
    h = key
    h = h ^ (h >> 16)
    h = h * 2146121005
    h = h ^ (h >> 15)
    h = h * -2073254261
    h = h ^ (h >> 16)
    bucket = wp.int32(h & (capacity - 1))
    step = wp.int32(0)
    while step < capacity:
        old = wp.atomic_cas(states, bucket, 0, 1)
        if old == 0:
            table_keys[bucket] = key
            table_values[bucket] = value
            return
        if table_keys[bucket] == key:
            table_values[bucket] = value
            return
        step = step + 1
        bucket = (bucket + 1) & (capacity - 1)


@wp.kernel
def clear_reduce_kernel(out: wp.array(dtype=wp.int64)):
    tid = wp.tid()
    if tid < 3:
        out[tid] = wp.int64(0)


@wp.kernel
def reduce_table_kernel(
    states: wp.array(dtype=wp.int32),
    table_keys: wp.array(dtype=wp.int32),
    table_values: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int64),
    capacity: int,
):
    tid = wp.tid()
    if tid >= capacity:
        return
    if states[tid] != 0:
        wp.atomic_add(out, 0, wp.int64(1))
        wp.atomic_add(out, 1, wp.int64(table_keys[tid]))
        wp.atomic_add(out, 2, wp.int64(table_values[tid]))


def next_power_of_two(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def capacity_for(entries: int) -> int:
    return next_power_of_two(max(1, entries * 2))


def sync(device: str) -> None:
    wp.synchronize_device(device)


def storage_model(entries: int, capacity: int) -> dict[str, int | str]:
    input_bytes = entries * 4 * 2
    table_bytes = capacity * 4 * 3
    output_bytes = 3 * 8
    return {
        "schema_version": 1,
        "source": "warp_open_addressing_hash_table",
        "entries": int(entries),
        "capacity": int(capacity),
        "input_bytes": int(input_bytes),
        "table_bytes": int(table_bytes),
        "output_bytes": int(output_bytes),
        "total_bytes": int(input_bytes + table_bytes + output_bytes),
    }


def run_hash(
    *,
    device: str,
    entries: int,
    fill,
    fill_inputs: list,
    steps: int,
    warmup: int,
    batch: int,
    sample_memory,
) -> dict:
    wp.init()
    if device == "cuda" and not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA unavailable")
    warp_device = "cuda:0" if device == "cuda" else "cpu"
    capacity = capacity_for(entries)
    input_keys = wp.zeros(entries, dtype=wp.int32, device=warp_device)
    input_values = wp.zeros(entries, dtype=wp.int32, device=warp_device)
    states = wp.zeros(capacity, dtype=wp.int32, device=warp_device)
    table_keys = wp.zeros(capacity, dtype=wp.int32, device=warp_device)
    table_values = wp.zeros(capacity, dtype=wp.int32, device=warp_device)
    out = wp.zeros(3, dtype=wp.int64, device=warp_device)

    def write_once():
        wp.launch(
            fill,
            dim=entries,
            inputs=[input_keys, input_values, entries] + fill_inputs,
            device=warp_device,
        )
        wp.launch(
            hash_insert_kernel,
            dim=entries,
            inputs=[
                input_keys,
                input_values,
                states,
                table_keys,
                table_values,
                entries,
                capacity,
            ],
            device=warp_device,
        )

    def reduce_once():
        wp.launch(clear_reduce_kernel, dim=3, inputs=[out], device=warp_device)
        wp.launch(
            reduce_table_kernel,
            dim=capacity,
            inputs=[states, table_keys, table_values, out, capacity],
            device=warp_device,
        )

    t0 = time.perf_counter()
    write_once()
    reduce_once()
    sync(warp_device)
    compile_first_s = time.perf_counter() - t0
    first_memory = sample_memory()
    first_out = out.numpy()

    for _ in range(warmup):
        write_once()
    sync(warp_device)
    write_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            write_once()
        sync(warp_device)
        write_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    for _ in range(warmup):
        reduce_once()
    sync(warp_device)
    reduce_samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for _ in range(batch):
            reduce_once()
        sync(warp_device)
        reduce_samples.append((time.perf_counter() - t0) * 1000.0 / batch)

    reduce_once()
    sync(warp_device)
    final_out = out.numpy()
    return {
        "compile_first_s": compile_first_s,
        "first_raw": [int(v) for v in first_out],
        "raw": [int(v) for v in final_out],
        "write_samples": write_samples,
        "reduce_samples": reduce_samples,
        "memory": {
            "after_first": first_memory,
            "after_bench": sample_memory(),
        },
        "external_memory_model": storage_model(entries, capacity),
    }


def run_root(device: str, active: int, domain: int, steps: int, warmup: int, batch: int, sample_memory) -> dict:
    return run_hash(
        device=device,
        entries=active,
        fill=fill_root_kernel,
        fill_inputs=[domain],
        steps=steps,
        warmup=warmup,
        batch=batch,
        sample_memory=sample_memory,
    )


def run_topology_2d(
    device: str,
    active: int,
    domain: int,
    inner_active: int,
    inner_domain: int,
    steps: int,
    warmup: int,
    batch: int,
    sample_memory,
) -> dict:
    return run_hash(
        device=device,
        entries=active * inner_active,
        fill=fill_topology_2d_kernel,
        fill_inputs=[active, domain, inner_active, inner_domain],
        steps=steps,
        warmup=warmup,
        batch=batch,
        sample_memory=sample_memory,
    )
