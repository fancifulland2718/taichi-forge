import argparse
import json
import math
import os
import unittest

from benchmarks.qualification.single_kernel_microbench import (
    _ExclusiveBenchmarkLock,
    QUALIFICATION_MINIMUMS,
    _enhanced_memory_plateau,
    _native_reduce_route,
    _native_transform_route,
    _native_indexed_copy_route,
    _native_compact_route,
    _device_prefix_chain_route,
    _particle_hash_route,
    _adaptive_pbd_route,
    _bfs_worklist_route,
    balanced_pair_orders,
    comparison_definition,
    comparison_participants,
    paired_log_summary,
    qualification_policy_errors,
    select_common_batch,
)
from benchmarks.qualification.runtime_common import normalize_gpu_uuid


class SingleKernelMicrobenchTest(unittest.TestCase):

    @unittest.skipUnless(os.name == "nt", "Windows named mutex test")
    def test_exclusive_driver_lock_rejects_overlap(self):
        with _ExclusiveBenchmarkLock():
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with _ExclusiveBenchmarkLock():
                    self.fail("overlapping lock was unexpectedly acquired")

    def test_balanced_pair_orders_are_adjacent_and_alternating(self):
        orders = balanced_pair_orders(5, 20260810)
        self.assertEqual(len(orders), 5)
        self.assertTrue(all(set(order) == {"forge", "vanilla"}
                            for order in orders))
        for left, right in zip(orders, orders[1:]):
            self.assertEqual(left, tuple(reversed(right)))

    def test_forge_native_control_uses_two_routes_from_forge(self):
        participants = comparison_participants(
            "forge-native-vs-forge-kernel", "native_transform")
        self.assertEqual(participants, ("forge", "forge_kernel"))
        orders = balanced_pair_orders(4, 20260810, participants)
        self.assertEqual(
            {tuple(order) for order in orders},
            {("forge", "forge_kernel"), ("forge_kernel", "forge")},
        )
        definition = comparison_definition(
            "forge-native-vs-forge-kernel", "native_transform")
        self.assertEqual(definition["speedup_formula"],
                         "forge_kernel_ms / forge_ms")
        self.assertEqual(definition["values_above_one_favor"], "forge")

    def test_forge_native_control_rejects_non_thin_operation(self):
        with self.assertRaisesRegex(ValueError, "thin/native operations"):
            comparison_participants(
                "forge-native-vs-forge-kernel", "mpm_graph")

    def test_kernel_compatibility_axis_uses_same_adapter_across_packages(self):
        definition = comparison_definition(
            "forge-kernel-vs-vanilla", "native_reduce")
        self.assertEqual(
            (definition["subject"], definition["baseline"]),
            ("forge_kernel", "vanilla_kernel"),
        )
        self.assertEqual(definition["speedup_formula"],
                         "vanilla_kernel_ms / forge_kernel_ms")
        self.assertIn("same vanilla-compatible kernel",
                      definition["attribution"])

    def test_forge_kernel_route_is_not_labeled_vanilla(self):
        route = _native_transform_route(None, "forge_kernel", "cuda")
        self.assertTrue(route["passed"])
        self.assertEqual(
            route["classification"],
            "forge_kernel_equivalent_i32_affine_kernel",
        )

    def test_common_batch_uses_larger_pilot_suggestion(self):
        self.assertEqual(select_common_batch([128, 512]), 512)
        with self.assertRaises(ValueError):
            select_common_batch([128, 0])

    def test_paired_log_summary_uses_pair_ratios(self):
        summary = paired_log_summary([2.0, 2.0, 2.0], seed=1, resamples=100)
        self.assertAlmostEqual(summary["median_speedup_x"], 2.0)
        self.assertAlmostEqual(summary["bootstrap_95_low_x"], 2.0)
        self.assertAlmostEqual(summary["bootstrap_95_high_x"], 2.0)
        self.assertTrue(math.isfinite(summary["median_speedup_x"]))

    def test_qualification_minimums_are_encoded_once(self):
        values = dict(QUALIFICATION_MINIMUMS)
        values.update(
            intent="qualification",
            backend="cuda",
            cpu_affinity="auto",
            max_cpu_util=20.0,
            max_gpu_util=15.0,
            max_gpu_temp=65.0,
        )
        args = argparse.Namespace(**values)
        self.assertEqual(qualification_policy_errors(args), [])
        args.pairs -= 1
        self.assertEqual(len(qualification_policy_errors(args)), 2)
        args.pairs = QUALIFICATION_MINIMUMS["pairs"] + 1
        self.assertEqual(qualification_policy_errors(args), [
            "qualification pairs must be even for exact AB/BA balance"
        ])
        args.intent = "diagnostic"
        self.assertEqual(qualification_policy_errors(args), [])

    def test_gpu_uuid_normalization_matches_runtime_and_nvidia_forms(self):
        self.assertEqual(
            normalize_gpu_uuid("GPU-a69ec138-0e30-8d1e-e299-4a0f2a2a6645"),
            "a69ec1380e308d1ee2994a0f2a2a6645",
        )
        self.assertEqual(
            normalize_gpu_uuid(bytes.fromhex(
                "a69ec1380e308d1ee2994a0f2a2a6645")),
            "a69ec1380e308d1ee2994a0f2a2a6645",
        )
        self.assertIsNone(normalize_gpu_uuid("not-a-device-uuid"))

    def test_enhanced_memory_plateau_rejects_live_growth(self):
        before = {
            "available": True,
            "runtime": {"memory": {
                "device_raw_bytes": 128,
                "device_requested_live_bytes": 64,
                "live_resources": 1,
            }},
            "pools": {
                "host": {"capacity_bytes": 0, "raw_bytes": 0,
                         "requested_live_bytes": 0, "reserved_bytes": 0,
                         "used_bytes": 0},
                "device": {"cached_blocks": 0, "cached_bytes": 0,
                           "raw_bytes": 128, "raw_chunks": 1},
            },
        }
        stable = json.loads(json.dumps(before))
        self.assertTrue(_enhanced_memory_plateau(before, stable)["passed"])
        growing = json.loads(json.dumps(before))
        growing["runtime"]["memory"]["live_resources"] = 2
        result = _enhanced_memory_plateau(before, growing)
        self.assertFalse(result["passed"])
        self.assertIn("live_resources",
                      result["runtime_memory"]["growing_fields"])

    def test_native_reduce_route_requires_the_declared_cuda_plan(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_reduce_ndarray"

        class Workspace:
            _native_reduce_plan = Plan()
            workspace_bytes_current = 4096
            workspace_bytes_peak = 4096

        passed = _native_reduce_route(Workspace(), "forge", "cuda")
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["observed_method"],
                         "cuda_device_reduce_ndarray")
        Plan.method_name = "unexpected_fallback"
        failed = _native_reduce_route(Workspace(), "forge", "cuda")
        self.assertFalse(failed["passed"])

    def test_native_transform_route_rejects_fallback(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_transform_affine_ndarray"

        class Workspace:
            _native_transform_plan = Plan()
            workspace_bytes_current = 0
            workspace_bytes_peak = 0

        self.assertTrue(
            _native_transform_route(Workspace(), "forge", "cuda")["passed"])
        Plan.backend = "field_kernel"
        self.assertFalse(
            _native_transform_route(Workspace(), "forge", "cuda")["passed"])

    def test_native_gather_route_requires_cached_native_plan(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_gather_ndarray"

        class Workspace:
            _native_indexed_copy_plan = Plan()
            workspace_bytes_current = 0
            workspace_bytes_peak = 0

        route = _native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=False)
        self.assertTrue(route["passed"])
        self.assertEqual(route["observed_method"],
                         "cuda_device_gather_ndarray")
        Plan.method_name = "gather_i32_ndarray_kernel_fallback"
        self.assertFalse(_native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=False)["passed"])

    def test_native_scatter_route_distinguishes_gather_plan(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_scatter_ndarray"

        class Workspace:
            _native_indexed_copy_plan = Plan()
            workspace_bytes_current = 0
            workspace_bytes_peak = 0

        self.assertTrue(_native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=True)["passed"])
        self.assertFalse(_native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=False)["passed"])

    def test_native_compact_route_requires_cuda_device_plan(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_compact_ndarray"

        class Workspace:
            _native_compact_plan = Plan()
            workspace_bytes_current = 8192
            workspace_bytes_peak = 8192

        self.assertTrue(
            _native_compact_route(Workspace(), "forge", "cuda")["passed"])
        Plan.backend = "cuda_cub"
        self.assertFalse(
            _native_compact_route(Workspace(), "forge", "cuda")["passed"])

    def test_device_prefix_route_requires_both_native_stages(self):
        class CompactPlan:
            backend = "cuda_device"
            method_name = "cuda_device_compact_ndarray"

        class ScanPlan:
            backend = "cuda_device"
            method_name = "cuda_device_inclusive_scan_ndarray"

        class CompactWorkspace:
            _native_compact_plan = CompactPlan()

        class Scanner:
            _native_scan_plan = ScanPlan()

        class Workspace:
            _compact = CompactWorkspace()
            _scan_executors = {64: Scanner()}
            workspace_bytes_current = 1024
            workspace_bytes_peak = 1024
            allocation_count = 2

        self.assertTrue(_device_prefix_chain_route(
            Workspace(), "forge", "cuda", 64)["passed"])
        ScanPlan.method_name = "legacy_scan"
        self.assertFalse(_device_prefix_chain_route(
            Workspace(), "forge", "cuda", 64)["passed"])

    def test_particle_hash_route_requires_native_bucket_plan(self):
        class Plan:
            backend = "cuda_device_bucket_builder"
            method_name = "cuda_device_bucket_builder_dense_field"

        class Workspace:
            _native_bucket_builder_plan = Plan()
            workspace_bytes_current = 64
            workspace_bytes_peak = 128

        route = _particle_hash_route(
            Workspace(), "forge", "cuda", bins=16)
        self.assertTrue(route["passed"])
        Plan.method_name = "field_kernel_fallback"
        self.assertFalse(_particle_hash_route(
            Workspace(), "forge", "cuda", bins=16)["passed"])

    def test_adaptive_pbd_route_requires_fixed_native_worklist(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_compact_ndarray"

        class CompactWorkspace:
            _native_compact_plan = Plan()
            workspace_bytes_current = 64
            workspace_bytes_peak = 64

        class Workspace:
            _compact = CompactWorkspace()

        class Worklist:
            workspace = Workspace()
            capacity = 32

            @staticmethod
            def memory_report():
                return {"fixed_capacity": True,
                        "replay_allocation_count": 0}

        self.assertTrue(_adaptive_pbd_route(
            Worklist(), "forge", "cuda")["passed"])
        Plan.method_name = "kernel_fallback"
        self.assertFalse(_adaptive_pbd_route(
            Worklist(), "forge", "cuda")["passed"])

    def test_bfs_route_rejects_overflow(self):
        class Stats:
            generated = 4
            accepted = 4
            rejected = 0
            overflow = False

        class Worklist:
            capacity = 32

            @staticmethod
            def memory_report():
                return {"fixed_capacity": True,
                        "replay_allocation_count": 0}

            @staticmethod
            def statistics():
                return Stats()

        self.assertTrue(_bfs_worklist_route(
            Worklist(), "forge", "cuda")["passed"])
        Stats.overflow = True
        self.assertFalse(_bfs_worklist_route(
            Worklist(), "forge", "cuda")["passed"])


if __name__ == "__main__":
    unittest.main()
