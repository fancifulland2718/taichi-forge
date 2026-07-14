"""Fresh-process dense Field multi-block compile and async replay benchmark.

The workload models the supported heterogeneous-environment architecture:
each block owns a distinct dense Field layout and Graph specialization, while
the leading Field dimension batches homogeneous environments inside a block.
Different blocks never share writable Fields.  The --display option adds a
concurrent Graph over an immutable snapshot and a private framebuffer; it
deliberately does not pretend that Graph supplies cross-stream hazard tracking.

The child report keeps definition/materialization, Graph dispatch compilation,
Graph finalization, first execution, steady completion, RSS/VRAM, and reset
measurements separate.  Parent mode always launches fresh child processes and
reports trial ranges; ranges above five percent are not suitable for formal
performance claims.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "DENSE_FIELD_MULTIBLOCK_RESULT "
SCHEMA = "taichi_forge.graph_dense_field_multiblock.v1"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(len(ordered) * fraction + 0.999999) - 1,
        ),
    )
    return ordered[index]


def _stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "sample": values,
    }


def _optional_median(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return statistics.median(available) if available else None


def _rss_mb() -> float | None:
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _gpu_process_mb(pid: int) -> float | None:
    if platform.system() == "Windows":
        command = (
            "$p="
            + str(pid)
            + ";$s=0;(Get-Counter "
            + "'\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples|"
            + "? InstanceName -like ('pid_'+$p+'_*')|%{$s+=$_.CookedValue};"
            + "[Console]::WriteLine([math]::Round($s/1MB,3))"
        )
        argv = ["powershell", "-NoProfile", "-Command", command]
    else:
        argv = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    try:
        output = subprocess.check_output(
            argv,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
        if platform.system() == "Windows":
            return float(output.strip())
        total = 0.0
        found = False
        for line in output.splitlines():
            process_id, used = [part.strip() for part in line.split(",", 1)]
            if int(process_id) == pid:
                total += float(used)
                found = True
        return total if found else None
    except Exception:
        return None


def _driver_version() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).splitlines()[0].strip()
    except Exception:
        return None


def _environment_preflight() -> dict[str, object]:
    gpu_state = None
    compute_processes = []
    try:
        gpu_state = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).strip()
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
        compute_processes = [
            line.strip() for line in output.splitlines() if line.strip()
        ]
    except Exception:
        pass
    other_python_processes = []
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        for process in psutil.process_iter(("pid", "name")):
            name = str(process.info.get("name") or "")
            if (
                process.info["pid"] != os.getpid()
                and "python" in name.lower()
            ):
                other_python_processes.append(
                    {
                        "pid": process.info["pid"],
                        "name": name,
                    }
                )
    except Exception:
        pass
    return {
        "gpu_utilization_memory_raw": gpu_state,
        "compute_processes": compute_processes,
        "other_python_processes": other_python_processes,
        "policy": (
            "desktop/GUI work is recorded but only compute processes "
            "invalidate a formal run"
        ),
    }


def _directory_state(path: str | None) -> dict[str, int] | None:
    if not path:
        return None
    root = Path(path)
    if not root.exists():
        return {"files": 0, "bytes": 0}
    files = [item for item in root.rglob("*") if item.is_file()]
    return {
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def _delta(after: float | None, before: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _arch_value(ti, name: str):
    return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def _block_items(base_items: int, index: int) -> int:
    return (
        base_items
        + (index % 4) * max(1, base_items // 2)
        + (index // 4) * max(1, base_items // 4)
    )


def _make_block_type(ti):
    @ti.data_oriented
    class DenseBlock:
        def __init__(self, env_count: int, item_count: int, variant: int):
            self.env_count = env_count
            self.item_count = item_count
            self.variant = variant
            self.position = ti.Vector.field(3, ti.f32)
            self.velocity = ti.Vector.field(3, ti.f32)
            self.mass = ti.field(ti.f32)
            self.domain = ti.field(ti.f32)
            self.counter = ti.field(ti.i32)
            self.snapshot = ti.Vector.field(3, ti.f32)
            self.display = ti.Vector.field(3, ti.f32)
            self.tree = None

        def materialize(self):
            builder = ti.FieldsBuilder()
            payload = builder.dense(
                ti.ij,
                (self.env_count, self.item_count),
            )
            payload.place(
                self.position,
                self.velocity,
                self.mass,
                self.snapshot,
                self.display,
            )
            per_env = builder.dense(ti.i, self.env_count)
            per_env.place(self.domain, self.counter)
            self.tree = builder.finalize()

        @ti.kernel
        def initialize(self):
            for env in self.domain:
                self.domain[env] = 0.75 + 0.01 * env
                self.counter[env] = 0
            for env, item in self.position:
                value = ti.cast((item + 3 * env) % 127, ti.f32) * 0.0005
                self.position[env, item] = ti.Vector(
                    [value, -0.5 * value, 0.25 * value]
                )
                self.velocity[env, item] = ti.Vector(
                    [0.001, -0.002, 0.0005]
                )
                self.mass[env, item] = 1.0 + 0.001 * (item % 31)
                self.snapshot[env, item] = ti.Vector.zero(ti.f32, 3)
                self.display[env, item] = ti.Vector.zero(ti.f32, 3)

        @ti.kernel
        def advance(self):
            for env in self.counter:
                self.counter[env] += 1
            for env, item in self.position:
                force = self.domain[env] * self.mass[env, item]
                self.velocity[env, item] += ti.Vector(
                    [force * 1e-6, -force * 2e-6, force * 5e-7]
                )
                self.position[env, item] += self.velocity[env, item] * 0.002

        @ti.kernel
        def damp(self):
            factor = 0.9998 - ti.static(self.variant) * 0.00001
            for env, item in self.velocity:
                self.velocity[env, item] *= factor

        @ti.kernel
        def constrain(self):
            bound = 0.2 + 0.01 * ti.static(self.variant)
            for env, item in self.position:
                value = self.position[env, item]
                for axis in ti.static(range(3)):
                    value[axis] = ti.max(-bound, ti.min(bound, value[axis]))
                self.position[env, item] = value

        @ti.kernel
        def couple(self):
            coefficient = 1e-7 * (ti.static(self.variant) + 1)
            for env, item in self.velocity:
                self.velocity[env, item] += (
                    self.position[env, item]
                    * self.domain[env]
                    * coefficient
                )

        @ti.kernel
        def publish(self):
            for env, item in self.snapshot:
                self.snapshot[env, item] = self.position[env, item]

        @ti.kernel
        def shade(self):
            gain = 0.8 + 0.01 * ti.static(self.variant)
            for env, item in self.display:
                self.display[env, item] = self.snapshot[env, item] * gain

        def sequence(self):
            variants = (
                (self.advance, self.damp, self.publish, self.shade),
                (
                    self.advance,
                    self.constrain,
                    self.damp,
                    self.publish,
                    self.shade,
                ),
                (
                    self.advance,
                    self.damp,
                    self.couple,
                    self.constrain,
                    self.publish,
                    self.shade,
                ),
                (
                    self.advance,
                    self.couple,
                    self.damp,
                    self.constrain,
                    self.couple,
                    self.publish,
                    self.shade,
                ),
            )
            return variants[self.variant % len(variants)]

        def payload_bytes(self) -> int:
            items = self.env_count * self.item_count
            vector_fields = 4
            return (
                items * vector_fields * 3 * 4
                + items * 4
                + self.env_count * 4
                + self.env_count * 4
            )

    return DenseBlock


def _make_display_type(ti):
    @ti.data_oriented
    class ImmutableSnapshotDisplay:
        def __init__(self, env_count: int, item_count: int):
            self.env_count = env_count
            self.item_count = item_count
            self.snapshot = ti.Vector.field(3, ti.f32)
            self.framebuffer = ti.Vector.field(3, ti.f32)
            self.tree = None

        def materialize(self):
            builder = ti.FieldsBuilder()
            builder.dense(
                ti.ij,
                (self.env_count, self.item_count),
            ).place(self.snapshot, self.framebuffer)
            self.tree = builder.finalize()

        @ti.kernel
        def initialize(self):
            for env, item in self.snapshot:
                value = ti.cast((env + item) % 97, ti.f32) * 0.001
                self.snapshot[env, item] = ti.Vector(
                    [value, value * 0.5, 1.0 - value]
                )
                self.framebuffer[env, item] = ti.Vector.zero(ti.f32, 3)

        @ti.kernel
        def tone_map(self):
            for env, item in self.framebuffer:
                value = self.snapshot[env, item]
                self.framebuffer[env, item] = value / (1.0 + value)

        @ti.kernel
        def overlay(self):
            for env, item in self.framebuffer:
                self.framebuffer[env, item] += ti.Vector(
                    [0.001, 0.0, 0.002]
                )

        def sequence(self):
            return (self.tone_map, self.overlay)

        def payload_bytes(self) -> int:
            return self.env_count * self.item_count * 2 * 3 * 4

    return ImmutableSnapshotDisplay


def _compile_owner_graph(ti, owner, diagnostics: bool):
    builder = ti.graph.GraphBuilder()
    dispatch_start = time.perf_counter()
    for kernel in owner.sequence():
        builder.dispatch(kernel, template_args={"self": owner})
    dispatch_compile_ms = (time.perf_counter() - dispatch_start) * 1000.0
    finalize_start = time.perf_counter()
    graph = builder.compile()
    graph_finalize_ms = (time.perf_counter() - finalize_start) * 1000.0
    if diagnostics:
        graph.execution_stats()
    return graph, dispatch_compile_ms, graph_finalize_ms


def _precompile_owner_kernels(ti, owners):
    """Measure frontend specialization and backend compilation separately."""

    compiled = []
    seen = set()
    frontend_start = time.perf_counter()
    for owner in owners:
        for kernel in (owner.initialize,) + tuple(owner.sequence()):
            primal = kernel._primal
            key = primal.ensure_compiled(owner)
            identity = (id(primal), key)
            if identity in seen:
                continue
            seen.add(identity)
            compiled.append(primal.compiled_kernels[key])
    frontend_ms = (time.perf_counter() - frontend_start) * 1000.0

    runtime = ti.lang.impl.get_runtime()
    program = runtime.prog
    backend_start = time.perf_counter()
    for kernel_cpp in compiled:
        program.compile_kernel(
            program.config(),
            program.get_device_caps(),
            kernel_cpp,
        )
    backend_ms = (time.perf_counter() - backend_start) * 1000.0
    return {
        "python_ast_specialization_ms": frontend_ms,
        "backend_compile_ms": backend_ms,
        "compiled_kernel_count": len(compiled),
    }


def _kernel_specialization_count(owners) -> int:
    primals = {}
    for owner in owners:
        for kernel in (owner.initialize,) + tuple(owner.sequence()):
            primal = kernel._primal
            primals[id(primal)] = primal
    return sum(len(primal.compiled_kernels) for primal in primals.values())


def _invoke_direct(owner) -> None:
    for kernel in owner.sequence():
        kernel()


def _run_sequential(ti, operations, repeats: int):
    round_ms = []
    submit_ms = [[] for _ in operations]
    start_all = time.perf_counter()
    for _ in range(repeats):
        start_round = time.perf_counter()
        for index, operation in enumerate(operations):
            start = time.perf_counter()
            operation()
            submit_ms[index].append(
                (time.perf_counter() - start) * 1000.0
            )
        ti.sync()
        round_ms.append((time.perf_counter() - start_round) * 1000.0)
    wall_seconds = time.perf_counter() - start_all
    rates = [
        repeats / max(1e-12, sum(samples) / 1000.0)
        for samples in submit_ms
    ]
    return {
        "wall_seconds": wall_seconds,
        "round_ms": _stats(round_ms),
        "submit_ms": [_stats(samples) for samples in submit_ms],
        "producer_host_rates": rates,
    }


def _run_concurrent(ti, operations, repeats: int):
    barrier = threading.Barrier(len(operations))
    samples = [[] for _ in operations]
    producer_seconds = [0.0 for _ in operations]
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(index: int, operation) -> None:
        try:
            barrier.wait(timeout=20.0)
            begin = time.perf_counter()
            for _ in range(repeats):
                start = time.perf_counter()
                operation()
                samples[index].append(
                    (time.perf_counter() - start) * 1000.0
                )
            producer_seconds[index] = time.perf_counter() - begin
        except BaseException as exc:
            with failure_lock:
                failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(index, operation))
        for index, operation in enumerate(operations)
    ]
    wall_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180.0)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("multi-block producer threads deadlocked")
    if failures:
        raise failures[0]
    ti.sync()
    wall_seconds = time.perf_counter() - wall_start
    rates = [
        repeats / max(1e-12, elapsed) for elapsed in producer_seconds
    ]
    return {
        "wall_seconds": wall_seconds,
        "round_ms": {"samples": 0},
        "submit_ms": [_stats(values) for values in samples],
        "producer_host_rates": rates,
    }


def _run_child(args) -> dict[str, object]:
    rss_process_start = _rss_mb()
    import_start = time.perf_counter()
    import numpy as np  # pylint: disable=import-outside-toplevel
    import taichi_forge as ti  # pylint: disable=import-outside-toplevel

    import_ms = (time.perf_counter() - import_start) * 1000.0
    init_kwargs = {
        "arch": _arch_value(ti, args.arch),
        "enable_fallback": False,
        "offline_cache": args.offline_cache,
    }
    if args.cache_dir:
        init_kwargs["offline_cache_file_path"] = args.cache_dir
    cache_before = _directory_state(args.cache_dir)
    init_start = time.perf_counter()
    try:
        ti.init(**init_kwargs)
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "skipped": True,
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "arch": args.arch,
        }
    init_ms = (time.perf_counter() - init_start) * 1000.0
    rss_before_fields = _rss_mb()
    sample_gpu = args.sample_gpu_memory and args.arch != "cpu"
    gpu_before_fields = _gpu_process_mb(os.getpid()) if sample_gpu else None

    DenseBlock = _make_block_type(ti)
    field_start = time.perf_counter()
    blocks = [
        DenseBlock(
            args.envs,
            _block_items(args.base_items, index),
            index % 4,
        )
        for index in range(args.blocks)
    ]
    display_owner = None
    if args.display:
        DisplayOwner = _make_display_type(ti)
        display_owner = DisplayOwner(
            args.envs,
            _block_items(args.base_items, 0),
        )
    field_definition_ms = (time.perf_counter() - field_start) * 1000.0

    owners = blocks + (
        [display_owner] if display_owner is not None else []
    )
    materialize_start = time.perf_counter()
    for owner in owners:
        owner.materialize()
    materialize_ms = (time.perf_counter() - materialize_start) * 1000.0
    rss_after_fields = _rss_mb()
    gpu_after_fields = _gpu_process_mb(os.getpid()) if sample_gpu else None

    compile_timings = _precompile_owner_kernels(ti, owners)
    rss_after_kernel_compile = _rss_mb()
    gpu_after_kernel_compile = (
        _gpu_process_mb(os.getpid()) if sample_gpu else None
    )

    initialize_start = time.perf_counter()
    for block in blocks:
        block.initialize()
    if display_owner is not None:
        display_owner.initialize()
    ti.sync()
    initialization_run_ms = (
        time.perf_counter() - initialize_start
    ) * 1000.0

    graphs = []
    graph_dispatch_compile_ms = []
    graph_finalize_ms = []
    if args.mode == "graph":
        for block in blocks:
            graph, dispatch_ms, finalize_ms = _compile_owner_graph(
                ti,
                block,
                args.diagnostics,
            )
            graphs.append(graph)
            graph_dispatch_compile_ms.append(dispatch_ms)
            graph_finalize_ms.append(finalize_ms)
        display_graph = None
        if display_owner is not None:
            display_graph, dispatch_ms, finalize_ms = _compile_owner_graph(
                ti,
                display_owner,
                args.diagnostics,
            )
            graph_dispatch_compile_ms.append(dispatch_ms)
            graph_finalize_ms.append(finalize_ms)
    else:
        display_graph = None
    rss_after_compile = _rss_mb()
    gpu_after_compile = _gpu_process_mb(os.getpid()) if sample_gpu else None

    if args.mode == "graph":
        block_operations = [
            (lambda graph=graph: graph.run({})) for graph in graphs
        ]
        display_operation = (
            (lambda: display_graph.run({}))
            if display_graph is not None
            else None
        )
    else:
        block_operations = [
            (lambda block=block: _invoke_direct(block)) for block in blocks
        ]
        display_operation = (
            (lambda: _invoke_direct(display_owner))
            if display_owner is not None
            else None
        )

    first_run_ms = []
    for operation in block_operations:
        first_start = time.perf_counter()
        operation()
        ti.sync()
        first_run_ms.append((time.perf_counter() - first_start) * 1000.0)
    display_first_run_ms = None
    if display_operation is not None:
        first_start = time.perf_counter()
        display_operation()
        ti.sync()
        display_first_run_ms = (
            time.perf_counter() - first_start
        ) * 1000.0
    rss_after_first = _rss_mb()
    gpu_after_first = _gpu_process_mb(os.getpid()) if sample_gpu else None

    for _ in range(args.warmups):
        for operation in block_operations:
            operation()
        if display_operation is not None:
            display_operation()
        ti.sync()

    operations = list(block_operations)
    if display_operation is not None:
        operations.append(display_operation)
    if args.submit == "sequential":
        steady = _run_sequential(ti, operations, args.repeats)
    else:
        steady = _run_concurrent(ti, operations, args.repeats)

    expected_counter = 1 + args.warmups + args.repeats
    for block in blocks:
        np.testing.assert_array_equal(
            block.counter.to_numpy(),
            np.full(args.envs, expected_counter, dtype=np.int32),
        )
        if not np.isfinite(block.display.to_numpy()).all():
            raise AssertionError("block display contains non-finite values")
    if display_owner is not None:
        if not np.isfinite(display_owner.framebuffer.to_numpy()).all():
            raise AssertionError("display framebuffer contains non-finite values")

    rss_after_steady = _rss_mb()
    gpu_after_steady = _gpu_process_mb(os.getpid()) if sample_gpu else None
    reports = (
        [asdict(graph.execution_stats()) for graph in graphs]
        if args.mode == "graph"
        else []
    )
    display_report = (
        asdict(display_graph.execution_stats())
        if display_graph is not None
        else None
    )
    report_set = reports + (
        [display_report] if display_report is not None else []
    )
    path_counts = Counter(
        report["execution_path"] for report in report_set
    )
    fallback_counts = Counter(
        report["fallback_reason"]
        for report in report_set
        if report["fallback_reason"] != "none"
    )
    persistent_argument_bytes = sum(
        segment["persistent_argument_bytes"]
        for report in report_set
        for segment in report["segments"]
    )
    compiled_task_count = sum(
        report["compiled_task_count"] for report in report_set
    )
    total_block_invocations = args.blocks * args.repeats
    total_env_steps = args.blocks * args.envs * args.repeats
    total_element_dispatches = sum(
        block.env_count
        * block.item_count
        * len(block.sequence())
        * args.repeats
        for block in blocks
    )
    wall_seconds = steady["wall_seconds"]
    producer_rates = steady["producer_host_rates"][: args.blocks]
    fairness = (
        min(producer_rates) / max(producer_rates)
        if producer_rates and max(producer_rates) > 0
        else 1.0
    )
    field_payload_bytes = sum(block.payload_bytes() for block in blocks)
    if display_owner is not None:
        field_payload_bytes += display_owner.payload_bytes()

    cache_after = _directory_state(args.cache_dir)
    result = {
        "schema": SCHEMA,
        "skipped": False,
        "arch": args.arch,
        "actual_arch": str(ti.lang.impl.current_cfg().arch),
        "mode": args.mode,
        "submit": args.submit,
        "blocks": args.blocks,
        "envs_per_block": args.envs,
        "base_items": args.base_items,
        "display": args.display,
        "diagnostics": args.diagnostics,
        "offline_cache": args.offline_cache,
        "cache_dir": args.cache_dir,
        "trial": args.trial,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "block_shapes": [
            [block.env_count, block.item_count] for block in blocks
        ],
        "block_dispatch_counts": [
            len(block.sequence()) for block in blocks
        ],
        "compile_phase_scope": {
            "field_definition_ms": "Python owner and Field declarations",
            "materialize_ms": "SNode tree materialization without kernel launch",
            "python_ast_specialization_ms": (
                "Kernel.ensure_compiled frontend/AST specialization; "
                "benchmark instrumentation only"
            ),
            "backend_compile_ms": (
                "Program.compile_kernel backend compilation/registration; "
                "benchmark instrumentation only"
            ),
            "initialization_run_ms": "warm initializer execution",
            "graph_dispatch_compile_ms": (
                "public GraphBuilder dispatch definition over precompiled "
                "kernel specializations"
            ),
            "graph_finalize_ms": (
                "compiled CGraph finalization over precompiled kernels"
            ),
            "first_run_ms": (
                "first executable launch plus CUDA capture/Vulkan record "
                "when eligible"
            ),
        },
        "timing_ms": {
            "import": import_ms,
            "init": init_ms,
            "field_definition": field_definition_ms,
            "materialize": materialize_ms,
            "python_ast_specialization": compile_timings[
                "python_ast_specialization_ms"
            ],
            "backend_compile": compile_timings["backend_compile_ms"],
            "initialization_run": initialization_run_ms,
            "graph_dispatch_compile_total": sum(
                graph_dispatch_compile_ms
            ),
            "graph_dispatch_compile_per_graph": graph_dispatch_compile_ms,
            "graph_finalize_total": sum(graph_finalize_ms),
            "graph_finalize_per_graph": graph_finalize_ms,
            "first_run_total": sum(first_run_ms)
            + (display_first_run_ms or 0.0),
            "first_run_per_block": first_run_ms,
            "display_first_run": display_first_run_ms,
        },
        "steady": {
            **steady,
            "completed_block_invocations_per_second": (
                total_block_invocations / wall_seconds
            ),
            "completed_env_steps_per_second": (
                total_env_steps / wall_seconds
            ),
            "completed_element_dispatches_per_second": (
                total_element_dispatches / wall_seconds
            ),
            "producer_fairness_min_over_max": fairness,
        },
        "specialization_count": _kernel_specialization_count(
            owners
        ),
        "precompiled_kernel_count": compile_timings[
            "compiled_kernel_count"
        ],
        "compiled_task_count": compiled_task_count,
        "graph_execution_reports": reports,
        "display_execution_report": display_report,
        "execution_path_counts": dict(path_counts),
        "fallback_reason_counts": dict(fallback_counts),
        "persistent_argument_bytes": persistent_argument_bytes,
        "field_payload_bytes": field_payload_bytes,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "cache_delta": (
            {
                "files": cache_after["files"] - cache_before["files"],
                "bytes": cache_after["bytes"] - cache_before["bytes"],
            }
            if cache_before is not None and cache_after is not None
            else None
        ),
        "rss_mb": {
            "process_start": rss_process_start,
            "before_fields": rss_before_fields,
            "after_fields": rss_after_fields,
            "after_kernel_compile": rss_after_kernel_compile,
            "after_compile": rss_after_compile,
            "after_first": rss_after_first,
            "after_steady": rss_after_steady,
            "field_delta": _delta(rss_after_fields, rss_before_fields),
            "kernel_compile_delta": _delta(
                rss_after_kernel_compile,
                rss_after_fields,
            ),
            "graph_compile_delta": _delta(
                rss_after_compile,
                rss_after_kernel_compile,
            ),
            "first_delta": _delta(rss_after_first, rss_after_compile),
            "steady_delta": _delta(rss_after_steady, rss_after_first),
        },
        "gpu_mb": {
            "before_fields": gpu_before_fields,
            "after_fields": gpu_after_fields,
            "after_kernel_compile": gpu_after_kernel_compile,
            "after_compile": gpu_after_compile,
            "after_first": gpu_after_first,
            "after_steady": gpu_after_steady,
            "field_delta": _delta(gpu_after_fields, gpu_before_fields),
            "kernel_compile_delta": _delta(
                gpu_after_kernel_compile,
                gpu_after_fields,
            ),
            "graph_compile_delta": _delta(
                gpu_after_compile,
                gpu_after_kernel_compile,
            ),
            "first_delta": _delta(gpu_after_first, gpu_after_compile),
            "steady_delta": _delta(gpu_after_steady, gpu_after_first),
        },
        "driver_version": _driver_version(),
        "python": sys.version,
        "platform": platform.platform(),
        "result": "pass",
    }

    reset_graphs = list(graphs)
    if display_graph is not None:
        reset_graphs.append(display_graph)
    ti.reset()
    result["reset_graphs_invalidated"] = (
        all(
            graph._spec is None
            and graph._instance is None
            and graph._instances == {}
            for graph in reset_graphs
        )
        if reset_graphs
        else None
    )
    reset_graphs.clear()
    graphs.clear()
    blocks.clear()
    owners.clear()
    operations.clear()
    block_operations.clear()
    display_graph = None
    display_owner = None
    display_operation = None
    gc.collect()
    time.sleep(0.05)
    result["rss_mb"]["after_reset"] = _rss_mb()
    result["rss_mb"]["reset_delta"] = _delta(
        result["rss_mb"]["after_reset"],
        rss_after_steady,
    )
    result["rss_mb"]["reset_over_pre_field_baseline"] = _delta(
        result["rss_mb"]["after_reset"],
        rss_before_fields,
    )
    result["gpu_mb"]["after_reset"] = (
        _gpu_process_mb(os.getpid()) if sample_gpu else None
    )
    result["gpu_mb"]["reset_delta"] = _delta(
        result["gpu_mb"]["after_reset"],
        gpu_after_steady,
    )
    result["gpu_mb"]["reset_over_pre_field_baseline"] = _delta(
        result["gpu_mb"]["after_reset"],
        gpu_before_fields,
    )
    result["cache_after_reset"] = _directory_state(args.cache_dir)
    result["cache_delta_after_reset"] = (
        {
            "files": (
                result["cache_after_reset"]["files"]
                - cache_before["files"]
            ),
            "bytes": (
                result["cache_after_reset"]["bytes"]
                - cache_before["bytes"]
            ),
        }
        if cache_before is not None
        and result["cache_after_reset"] is not None
        else None
    )
    return result


def _child_command(args, config, trial: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arch",
        config["arch"],
        "--mode",
        config["mode"],
        "--submit",
        config["submit"],
        "--blocks",
        str(config["blocks"]),
        "--envs",
        str(config["envs"]),
        "--base-items",
        str(args.base_items),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--trial",
        str(trial),
    ]
    if args.display:
        command.append("--display")
    if args.diagnostics:
        command.append("--diagnostics")
    if args.sample_gpu_memory:
        command.append("--sample-gpu-memory")
    if args.offline_cache:
        command.append("--offline-cache")
    if args.cache_dir:
        command.extend(["--cache-dir", args.cache_dir])
    return command


def _child_env(args, arch: str) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(ROOT / "python"), str(ROOT)]
    if args.pythonpath:
        paths.insert(0, str(Path(args.pythonpath).resolve()))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONIOENCODING"] = "utf-8"
    env["TAICHI_OFFLINE_CACHE"] = "1" if args.offline_cache else "0"
    env["TI_WANTED_ARCHS"] = arch
    return env


def _run_fresh(args, config, trial: int) -> dict[str, object]:
    process = subprocess.run(
        _child_command(args, config, trial),
        capture_output=True,
        text=True,
        env=_child_env(args, config["arch"]),
        timeout=args.timeout,
        check=False,
    )
    if args.verbose and process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    for line in process.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(
        "child did not emit a result "
        f"(exit={process.returncode}): {process.stdout[-2000:]}"
    )


def _configurations(args):
    arches = [item.strip() for item in args.arches.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if args.matrix:
        block_counts = (1, 2, 4, 8)
        env_counts = (1, 8, 32, 128)
        submit_modes = ("sequential", "concurrent")
    else:
        block_counts = (args.blocks,)
        env_counts = (args.envs,)
        submit_modes = (args.submit,)
    return [
        {
            "arch": arch,
            "mode": mode,
            "submit": submit,
            "blocks": blocks,
            "envs": envs,
        }
        for arch in arches
        for mode in modes
        for submit in submit_modes
        for blocks in block_counts
        for envs in env_counts
    ]


def _summaries(results):
    groups = {}
    for result in results:
        if result.get("skipped"):
            continue
        key = (
            result["arch"],
            result["mode"],
            result["submit"],
            result["blocks"],
            result["envs_per_block"],
            result["display"],
        )
        groups.setdefault(key, []).append(result)
    summaries = []
    for key, trials in groups.items():
        throughput = [
            item["steady"][
                "completed_block_invocations_per_second"
            ]
            for item in trials
        ]
        median = statistics.median(throughput)
        range_percent = (
            (max(throughput) - min(throughput)) / median * 100.0
            if median > 0
            else 0.0
        )
        summaries.append(
            {
                "arch": key[0],
                "mode": key[1],
                "submit": key[2],
                "blocks": key[3],
                "envs_per_block": key[4],
                "display": key[5],
                "trials": len(trials),
                "throughput_median": median,
                "throughput_range_percent": range_percent,
                "trial_count_sufficient": len(trials) >= 3,
                "formal_range_gate_pass": (
                    len(trials) >= 3 and range_percent <= 5.0
                ),
                "build_first_median_ms": statistics.median(
                    item["timing_ms"]["python_ast_specialization"]
                    + item["timing_ms"]["backend_compile"]
                    + item["timing_ms"]["graph_dispatch_compile_total"]
                    + item["timing_ms"]["graph_finalize_total"]
                    + item["timing_ms"]["first_run_total"]
                    for item in trials
                ),
                "rss_after_steady_median_mb": _optional_median(
                    [
                        item["rss_mb"]["after_steady"]
                        for item in trials
                    ]
                ),
                "gpu_after_steady_values_mb": [
                    item["gpu_mb"]["after_steady"] for item in trials
                ],
                "execution_path_counts": dict(
                    sum(
                        (
                            Counter(item["execution_path_counts"])
                            for item in trials
                        ),
                        Counter(),
                    )
                ),
            }
        )
    return summaries


def _mode_comparisons(summaries):
    groups = {}
    for item in summaries:
        key = (
            item["arch"],
            item["submit"],
            item["blocks"],
            item["envs_per_block"],
            item["display"],
        )
        groups.setdefault(key, {})[item["mode"]] = item
    comparisons = []
    for key, modes in groups.items():
        if "direct" not in modes or "graph" not in modes:
            continue
        direct = modes["direct"]
        graph = modes["graph"]
        ratio = graph["throughput_median"] / direct["throughput_median"]
        comparisons.append(
            {
                "arch": key[0],
                "submit": key[1],
                "blocks": key[2],
                "envs_per_block": key[3],
                "display": key[4],
                "graph_over_direct": ratio,
                "graph_throughput_change_percent": (ratio - 1.0) * 100.0,
                "formal_comparison": (
                    direct["formal_range_gate_pass"]
                    and graph["formal_range_gate_pass"]
                ),
            }
        )
    return comparisons


def _scaling_summaries(summaries):
    groups = {}
    for item in summaries:
        key = (
            item["arch"],
            item["mode"],
            item["submit"],
            item["envs_per_block"],
            item["display"],
        )
        groups.setdefault(key, {})[item["blocks"]] = item
    scaling = []
    for key, blocks in groups.items():
        if 1 not in blocks or 8 not in blocks:
            continue
        ratio = (
            blocks[8]["build_first_median_ms"]
            / blocks[1]["build_first_median_ms"]
        )
        scaling.append(
            {
                "arch": key[0],
                "mode": key[1],
                "submit": key[2],
                "envs_per_block": key[3],
                "display": key[4],
                "build_first_8_over_1": ratio,
                "build_first_normalized_per_block": ratio / 8.0,
                "unexplained_superlinear_guard": ratio / 8.0 <= 1.25,
            }
        )
    return scaling


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--arches",
        default="cpu",
        help="comma-separated parent-process architecture list",
    )
    parser.add_argument(
        "--modes",
        default="graph",
        help="comma-separated parent-process mode list",
    )
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("--mode", choices=("direct", "graph"), default="graph")
    parser.add_argument(
        "--submit",
        choices=("sequential", "concurrent"),
        default="sequential",
    )
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--base-items", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--sample-gpu-memory", action="store_true")
    parser.add_argument("--offline-cache", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--pythonpath")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    for name in ("blocks", "envs", "base_items", "repeats", "trials"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.child and args.arch is None:
        parser.error("--child requires --arch")
    return args


def main() -> None:
    args = _parse_args()
    if args.child:
        result = _run_child(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return

    preflight = _environment_preflight()
    results = []
    configurations = _configurations(args)
    for trial in range(args.trials):
        trial_configurations = (
            configurations
            if trial % 2 == 0
            else reversed(configurations)
        )
        for config in trial_configurations:
            results.append(_run_fresh(args, config, trial))
    summaries = _summaries(results)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment_preflight": preflight,
        "fresh_process_trials": args.trials,
        "range_policy": (
            "throughput_range_percent > 5 is observational only and must "
            "be rerun before a formal performance claim"
        ),
        "results": results,
        "summaries": summaries,
        "mode_comparisons": _mode_comparisons(summaries),
        "scaling_summaries": _scaling_summaries(summaries),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
