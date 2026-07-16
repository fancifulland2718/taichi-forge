"""Long-running native primitive, Graph, and workspace lifetime stress.

This is a correctness/lifetime stress, not a performance benchmark. It reports
iteration counts only as progress evidence and never derives throughput. Use
the guarded benchmark suite when collecting performance data.

Examples:

  python tests/python/native_primitive_runtime_stress.py --arch cpu --seconds 30
  python tests/python/native_primitive_runtime_stress.py --arch cuda --seconds 60
  python tests/python/native_primitive_runtime_stress.py --arch vulkan --seconds 60
"""

import argparse
import json
import threading
import time

import numpy as np

import taichi_forge as ti


class WorkerState:
    def __init__(self, items, num_bins, seed):
        self.items = items
        self.num_bins = num_bins
        self.seed = seed
        index = np.arange(items, dtype=np.int32)
        self.source_host = ((index + seed) % 97 - 48).astype(np.int32)
        self.flags_host = ((index + seed) % 3 != 0).astype(np.int32)
        self.histogram_host = ((index + seed) % num_bins).astype(np.int32)
        self.sort_keys_host = (((index + seed) * 17) % 257 - 128).astype(
            np.int32
        )
        self.sort_values_host = index.copy()
        order = np.argsort(self.sort_keys_host, kind="stable")
        self.expected_transform = self.source_host * 3 + 7
        self.expected_compact = self.expected_transform[
            self.flags_host.astype(bool)
        ]
        self.expected_histogram = np.bincount(
            self.histogram_host, minlength=num_bins
        ).astype(np.int32)
        self.expected_sort_keys = self.sort_keys_host[order]
        self.expected_sort_values = self.sort_values_host[order]

        self.source = ti.ndarray(ti.i32, shape=items)
        self.transformed = ti.ndarray(ti.i32, shape=items)
        self.flags = ti.ndarray(ti.i32, shape=items)
        self.compacted = ti.ndarray(ti.i32, shape=items)
        self.compact_count = ti.ndarray(ti.i32, shape=1)
        self.histogram_values = ti.ndarray(ti.i32, shape=items)
        self.histogram_bins = ti.ndarray(ti.i32, shape=num_bins)
        self.sort_keys = ti.ndarray(ti.i32, shape=items)
        self.sort_values = ti.ndarray(ti.i32, shape=items)
        self.source.from_numpy(self.source_host)
        self.flags.from_numpy(self.flags_host)
        self.histogram_values.from_numpy(self.histogram_host)
        self.sort_keys.from_numpy(self.sort_keys_host)
        self.sort_values.from_numpy(self.sort_values_host)

        self.transform_workspace = ti.algorithms.TransformWorkspace(
            max_items=items
        )
        self.compact_workspace = ti.algorithms.CompactWorkspace(max_items=items)
        self.histogram_workspace = ti.algorithms.HistogramWorkspace(
            max_items=items, max_bins=num_bins
        )
        self.sort_workspace = ti.algorithms.SortWorkspace(max_items=items)
        self.sequence = (
            ti.algorithms.primitive_sequence()
            .transform(
                self.source,
                self.transformed,
                scale=3,
                bias=7,
                workspace=self.transform_workspace,
            )
            .compact(
                self.transformed,
                self.flags,
                self.compacted,
                self.compact_count,
                workspace=self.compact_workspace,
            )
        )
        self.sequence.prewarm()
        builder = ti.graph.GraphBuilder()
        builder.append_native(self.sequence)
        self.graph = builder.compile()
        ti.algorithms.experimental_histogram(
            self.histogram_values,
            self.histogram_bins,
            workspace=self.histogram_workspace,
        )
        ti.algorithms.sort(
            self.sort_keys,
            self.sort_values,
            workspace=self.sort_workspace,
        )
        self.iterations = 0

    def run_once(self):
        self.graph.run({})
        ti.algorithms.experimental_histogram(
            self.histogram_values,
            self.histogram_bins,
            workspace=self.histogram_workspace,
        )
        ti.algorithms.sort(
            self.sort_keys,
            self.sort_values,
            workspace=self.sort_workspace,
        )

    def validate(self):
        np.testing.assert_array_equal(
            self.transformed.to_numpy(), self.expected_transform
        )
        count = int(self.compact_count.to_numpy()[0])
        assert count == self.expected_compact.size
        np.testing.assert_array_equal(
            self.compacted.to_numpy()[:count], self.expected_compact
        )
        np.testing.assert_array_equal(
            self.histogram_bins.to_numpy(), self.expected_histogram
        )
        np.testing.assert_array_equal(
            self.sort_keys.to_numpy(), self.expected_sort_keys
        )
        np.testing.assert_array_equal(
            self.sort_values.to_numpy(), self.expected_sort_values
        )

    def clear(self):
        self.sequence.clear()
        self.histogram_workspace.clear()
        self.sort_workspace.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--items", type=int, default=1 << 14)
    parser.add_argument("--bins", type=int, default=257)
    parser.add_argument("--sync-interval", type=int, default=16)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be positive")
    if args.threads <= 0 or args.items <= 0 or args.bins <= 0:
        raise ValueError("threads, items, and bins must be positive")
    if args.sync_interval <= 0:
        raise ValueError("sync-interval must be positive")

    ti.init(arch=getattr(ti, args.arch), cpu_max_num_threads=4)
    states = [
        WorkerState(args.items, args.bins, seed=worker * 19)
        for worker in range(args.threads)
    ]
    ti.sync()

    barrier = threading.Barrier(args.threads + 1)
    failure_lock = threading.Lock()
    failures = []
    deadline = {"value": 0.0}

    def run_worker(state):
        try:
            barrier.wait(timeout=20)
            while time.perf_counter() < deadline["value"]:
                state.run_once()
                state.iterations += 1
                if state.iterations % args.sync_interval == 0:
                    ti.sync()
        except BaseException as exc:
            with failure_lock:
                failures.append(exc)

    threads = [
        threading.Thread(target=run_worker, args=(state,))
        for state in states
    ]
    for thread in threads:
        thread.start()
    start = time.perf_counter()
    deadline["value"] = start + args.seconds
    barrier.wait(timeout=20)
    for thread in threads:
        thread.join(timeout=args.seconds + 120)
    wall_seconds = time.perf_counter() - start
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("native primitive runtime stress deadlocked")
    if failures:
        raise failures[0]
    if any(state.iterations <= 0 for state in states):
        raise RuntimeError("one or more stress workers made no progress")

    ti.sync()
    for state in states:
        state.validate()

    # Prove that a quiescent sequence clear rebuilds its native replay plans
    # without stale bindings, then isolate one diagnostic interval.
    for state in states:
        state.sequence.clear()
        state.graph.run({})
    ti.sync()
    for state in states:
        state.validate()

    ti.algorithms.set_primitive_diagnostics_enabled(True, clear=True)
    try:
        states[0].run_once()
        diagnostics = ti.algorithms.get_primitive_runtime_diagnostics()
    finally:
        ti.algorithms.set_primitive_diagnostics_enabled(False, clear=True)
    ti.sync()
    states[0].validate()
    workspace_before_clear = diagnostics["workspace"]
    provider_names = {
        provider["provider"] for provider in diagnostics["providers"]
    }
    if args.arch == "cuda":
        toolkit_providers = [
            provider
            for provider in diagnostics["providers"]
            if provider["dependency_class"] == "toolkit_reference"
        ]
        if toolkit_providers:
            raise RuntimeError(
                "automatic CUDA dispatch used Toolkit-reference providers: "
                f"{toolkit_providers}"
            )
        if any(
            name.startswith("cuda_cub_")
            for name in workspace_before_clear["program_provider_bytes"]
        ):
            raise RuntimeError(
                "CUDA workspace telemetry double-counted shared CUB aliases"
            )
    required_provider_tokens = ("transform", "compact", "histogram", "sort")
    missing_provider_tokens = [
        token
        for token in required_provider_tokens
        if not any(token in provider for provider in provider_names)
    ]
    if missing_provider_tokens:
        raise RuntimeError(
            "provider diagnostics missed executed primitive families: "
            f"{missing_provider_tokens}; providers={sorted(provider_names)}"
        )

    for state in states:
        state.clear()
    ti.algorithms.clear_default_workspaces()
    workspace_after_clear = ti.algorithms.get_primitive_workspace_statistics()
    if workspace_after_clear["program_provider_errors"]:
        raise RuntimeError(
            "workspace telemetry reported provider errors: "
            f"{workspace_after_clear['program_provider_errors']}"
        )
    if workspace_after_clear["default_cache"]["entry_count"] != 0:
        raise RuntimeError("default primitive workspace cache did not clear")
    if (
        workspace_after_clear["program_provider_bytes_total"]
        > workspace_before_clear["program_provider_bytes_total"]
    ):
        raise RuntimeError("Program primitive workspace grew after clear")

    report = {
        "arch": args.arch,
        "items": args.items,
        "threads": args.threads,
        "seconds_requested": args.seconds,
        "wall_seconds": round(wall_seconds, 3),
        "iterations_per_worker": [state.iterations for state in states],
        "providers": diagnostics["providers"],
        "fallbacks": diagnostics["fallbacks"],
        "workspace_before_clear": workspace_before_clear,
        "workspace_after_clear": workspace_after_clear,
        "performance": "not_measured",
        "result": "pass",
    }
    print(json.dumps(report, sort_keys=True))
    ti.reset()


if __name__ == "__main__":
    main()
