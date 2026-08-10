import argparse
import json
import math
import os
import unittest
from unittest.mock import patch

from benchmarks.qualification.single_kernel_microbench import (
    _ExclusiveBenchmarkLock,
    _calibrate_batch,
    QUALIFICATION_MINIMUMS,
    _enhanced_memory_plateau,
    _snode_lifecycle_plateau,
    _run_stability,
    _StabilityReplayError,
    _native_reduce_route,
    _native_transform_route,
    _native_indexed_copy_route,
    _native_compact_route,
    _ordinary_kernel_route,
    _endpoint_equivalent,
    _device_prefix_chain_route,
    _particle_hash_route,
    _adaptive_pbd_route,
    _bfs_worklist_route,
    balanced_pair_orders,
    comparison_definition,
    comparison_participants,
    main,
    paired_log_summary,
    qualification_policy_errors,
    select_common_batch,
    warmup_batch_size,
)
from benchmarks.qualification.runtime_common import normalize_gpu_uuid
from benchmarks.qualification.audit_single_kernel_run import (
    _endpoint_equivalent as _audit_endpoint_equivalent,
)


class SingleKernelMicrobenchTest(unittest.TestCase):

    def test_ordinary_kernel_route_proves_benchmark_owned_single_launch(self):
        route = _ordinary_kernel_route("forge", "a" * 64)
        self.assertTrue(route["passed"])
        self.assertEqual(route["classification"],
                         "forge_ordinary_taichi_kernel")
        self.assertEqual(route["adapter"], "direct_ti_kernel")
        self.assertFalse(route["native_or_helper_api_used"])
        self.assertEqual(route["ti_kernel_invocations_per_replay"], 1)
        self.assertFalse(route["physical_backend_launches_assumed"])
        self.assertFalse(_ordinary_kernel_route("forge", "short")["passed"])

    def test_ordinary_endpoint_equivalence_uses_actual_fingerprints(self):
        validation = {
            "passed": True,
            "effective_tolerance": 0.0,
            "endpoint_fingerprint": {
                "finite": True,
                "count": 4,
                "sum": 10.0,
                "minimum": 1.0,
                "maximum": 4.0,
                "sample_indices": [0, 1, 2, 3],
                "sample_values": [1.0, 2.0, 3.0, 4.0],
            },
        }
        results = {
            "forge": {
                "operation": "copy",
                "validation_before": json.loads(json.dumps(validation)),
                "validation_after": json.loads(json.dumps(validation)),
            },
            "vanilla": {
                "operation": "copy",
                "validation_before": json.loads(json.dumps(validation)),
                "validation_after": json.loads(json.dumps(validation)),
            },
        }
        self.assertTrue(_endpoint_equivalent(results, "forge", "vanilla"))
        self.assertTrue(_audit_endpoint_equivalent(
            results["forge"], results["vanilla"]))
        results["vanilla"]["validation_after"]["endpoint_fingerprint"][
            "sample_values"][2] = 3.5
        results["vanilla"]["validation_after"]["endpoint_fingerprint"][
            "sum"] = 10.5
        self.assertFalse(_endpoint_equivalent(results, "forge", "vanilla"))
        self.assertFalse(_audit_endpoint_equivalent(
            results["forge"], results["vanilla"]))

    def test_native_reduce_endpoint_equivalence_uses_exact_actual_values(self):
        validation = {
            "passed": True,
            "actual": -8,
            "expected": -8,
            "absolute_error": 0,
        }
        results = {
            "forge": {
                "operation": "native_reduce",
                "validation_before": dict(validation),
                "validation_after": dict(validation),
            },
            "forge_kernel": {
                "operation": "native_reduce",
                "validation_before": dict(validation),
                "validation_after": dict(validation),
            },
        }
        self.assertTrue(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertTrue(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))
        results["forge_kernel"]["validation_after"]["actual"] = -7
        self.assertFalse(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertFalse(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))

    def test_native_transform_endpoint_equivalence_uses_exact_fingerprint(self):
        validation = {
            "passed": True,
            "comparison": "exact_i32_affine_transform",
            "count": 4,
            "actual_sha256": "a" * 64,
            "expected_sha256": "a" * 64,
            "actual_sum": 40,
            "expected_sum": 40,
            "actual_minimum": 1,
            "expected_minimum": 1,
            "actual_maximum": 19,
            "expected_maximum": 19,
            "sample_indices": [0, 1, 2, 3],
            "actual_samples": [1, 7, 13, 19],
            "expected_samples": [1, 7, 13, 19],
            "mismatch_count": 0,
            "first_mismatch": None,
        }
        results = {
            name: {
                "operation": "native_transform",
                "validation_before": json.loads(json.dumps(validation)),
                "validation_after": json.loads(json.dumps(validation)),
            }
            for name in ("forge", "forge_kernel")
        }
        self.assertTrue(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertTrue(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))
        results["forge_kernel"]["validation_after"]["actual_sha256"] = "b" * 64
        self.assertFalse(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertFalse(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))

    def test_native_gather_endpoint_equivalence_uses_exact_fingerprint(self):
        validation = {
            "passed": True,
            "comparison": "exact_i32_gather",
            "count": 4,
            "actual_sha256": "c" * 64,
            "expected_sha256": "c" * 64,
            "actual_sum": 12,
            "expected_sum": 12,
            "actual_minimum": -3,
            "expected_minimum": -3,
            "actual_maximum": 8,
            "expected_maximum": 8,
            "sample_indices": [0, 1, 2, 3],
            "actual_samples": [-3, 2, 5, 8],
            "expected_samples": [-3, 2, 5, 8],
            "mismatch_count": 0,
            "first_mismatch": None,
        }
        results = {
            name: {
                "operation": "native_gather",
                "validation_before": json.loads(json.dumps(validation)),
                "validation_after": json.loads(json.dumps(validation)),
            }
            for name in ("forge", "forge_kernel")
        }
        self.assertTrue(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertTrue(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))
        results["forge_kernel"]["validation_after"]["actual_samples"][1] = 3
        self.assertFalse(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertFalse(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))

    def test_native_scatter_endpoint_equivalence_uses_exact_fingerprint(self):
        validation = {
            "passed": True,
            "comparison": "exact_i32_scatter",
            "count": 4,
            "actual_sha256": "d" * 64,
            "expected_sha256": "d" * 64,
            "actual_sum": 12,
            "expected_sum": 12,
            "actual_minimum": -3,
            "expected_minimum": -3,
            "actual_maximum": 8,
            "expected_maximum": 8,
            "sample_indices": [0, 1, 2, 3],
            "actual_samples": [5, -3, 8, 2],
            "expected_samples": [5, -3, 8, 2],
            "mismatch_count": 0,
            "first_mismatch": None,
        }
        results = {
            name: {
                "operation": "native_scatter",
                "validation_before": json.loads(json.dumps(validation)),
                "validation_after": json.loads(json.dumps(validation)),
            }
            for name in ("forge", "forge_kernel")
        }
        self.assertTrue(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertTrue(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))
        results["forge_kernel"]["validation_after"]["actual_sum"] = 13
        self.assertFalse(
            _endpoint_equivalent(results, "forge", "forge_kernel"))
        self.assertFalse(_audit_endpoint_equivalent(
            results["forge"], results["forge_kernel"]))

    def test_profiler_range_rejects_normal_parent_or_non_cuda_run(self):
        with self.assertRaisesRegex(
                ValueError, "requires one CUDA score sample in child mode"):
            main([
                "--operation", "fill",
                "--backend", "cpu",
                "--preset", "small",
                "--cuda-profiler-range",
            ])

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
        source_sha256 = "a" * 64
        route = _native_transform_route(
            None, "forge_kernel", "cuda", source_sha256)
        self.assertTrue(route["passed"])
        self.assertEqual(
            route["classification"],
            "forge_kernel_equivalent_i32_affine_kernel",
        )
        self.assertEqual(route["adapter"], "benchmark_defined_ti_kernel")
        self.assertEqual(route["kernel_source_owner"], "benchmark")
        self.assertEqual(route["kernel_source_sha256"], source_sha256)
        self.assertFalse(route["helper_api_used"])
        self.assertFalse(route["workspace_present"])
        self.assertEqual(route["ti_kernel_invocations_per_replay"], 1)
        self.assertFalse(route["physical_backend_launches_assumed"])
        self.assertFalse(_native_transform_route(
            object(), "forge_kernel", "cuda", source_sha256)["passed"])
        self.assertFalse(_native_transform_route(
            None, "forge_kernel", "cuda", "short")["passed"])

    def test_native_reduce_kernel_route_proves_benchmark_control(self):
        source_sha256 = "a" * 64
        route = _native_reduce_route(
            None, "forge_kernel", "cuda", source_sha256)
        self.assertTrue(route["passed"])
        self.assertEqual(route["adapter"], "benchmark_defined_ti_kernel")
        self.assertEqual(route["kernel_source_owner"], "benchmark")
        self.assertEqual(route["kernel_source_sha256"], source_sha256)
        self.assertFalse(route["helper_api_used"])
        self.assertFalse(route["workspace_present"])
        self.assertEqual(route["ti_kernel_invocations_per_replay"], 1)
        self.assertFalse(route["physical_backend_launches_assumed"])
        self.assertFalse(_native_reduce_route(
            object(), "forge_kernel", "cuda", source_sha256)["passed"])
        self.assertFalse(_native_reduce_route(
            None, "forge_kernel", "cuda", "short")["passed"])

    def test_common_batch_uses_larger_pilot_suggestion(self):
        self.assertEqual(select_common_batch([128, 512]), 512)
        with self.assertRaises(ValueError):
            select_common_batch([128, 0])

    def test_score_warmup_uses_frozen_common_batch(self):
        self.assertEqual(warmup_batch_size("pilot", 2216), 1)
        self.assertEqual(warmup_batch_size("score", 2216), 2216)
        with self.assertRaises(ValueError):
            warmup_batch_size("score", 0)
        with self.assertRaises(ValueError):
            warmup_batch_size("unknown", 1)

    def test_pilot_confirms_candidate_batch_after_steady_state(self):
        timings = iter([1.0, 130.0, 90.0, 90.0, 180.0, 170.0, 160.0])
        with patch(
                "benchmarks.qualification.single_kernel_microbench._timed_batch",
                side_effect=lambda *_: next(timings)):
            batch, attempts = _calibrate_batch(None, lambda: None, 120.0)
        self.assertEqual(batch, 240)
        self.assertEqual([row["batch_size"] for row in attempts],
                         [1, 120, 120, 120, 240, 240, 240])
        self.assertEqual(
            [row["confirmation"] for row in attempts],
            [False, False, True, True, False, True, True],
        )

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

    def test_snode_lifecycle_plateau_ignores_retired_shell_growth(self):
        before = {
            "required": True,
            "available": True,
            "runtime_directory": {
                "active_tree_count": 1,
                "capacity": 16,
                "reserved_bytes": 128,
                "growth_events": 1,
            },
            "kernel_lifecycle": {
                "total_slots": 2,
                "live_definitions": 2,
                "retired_shells": 0,
                "registered_executables": 2,
            },
            "snode_field_mapping_count": 1,
        }
        after = json.loads(json.dumps(before))
        after["kernel_lifecycle"]["total_slots"] = 202
        after["kernel_lifecycle"]["retired_shells"] = 200
        stable = _snode_lifecycle_plateau(before, after)
        self.assertTrue(stable["passed"])
        self.assertEqual(stable["kernel_deltas"]["retired_shells"], 200)
        growing = json.loads(json.dumps(after))
        growing["kernel_lifecycle"]["live_definitions"] = 3
        result = _snode_lifecycle_plateau(before, growing)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["live_definitions_recovered"])

    def test_snode_lifecycle_initial_capacity_growth_can_be_allowed(self):
        before = {
            "required": True,
            "available": True,
            "runtime_directory": {
                "active_tree_count": 1,
                "capacity": 16,
                "reserved_bytes": 128,
                "growth_events": 1,
            },
            "kernel_lifecycle": {
                "total_slots": 2,
                "live_definitions": 2,
                "retired_shells": 0,
                "registered_executables": 2,
            },
            "snode_field_mapping_count": 1,
        }
        after = json.loads(json.dumps(before))
        after["runtime_directory"].update(
            capacity=128, reserved_bytes=1024, growth_events=2)
        after["kernel_lifecycle"]["registered_executables"] = 3
        self.assertFalse(_snode_lifecycle_plateau(before, after)["passed"])
        self.assertTrue(_snode_lifecycle_plateau(
            before, after, require_directory_plateau=False,
            require_registration_plateau=False)["passed"])

    def test_stability_failure_preserves_partial_replay_and_memory_evidence(self):
        calls = 0

        def launch():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("synthetic lifecycle failure")

        class FakeTi:

            @staticmethod
            def sync():
                pass

        observation = {
            "available": False,
            "runtime": None,
            "pools": None,
            "runtime_error": None,
            "pool_error": None,
        }
        with patch(
                "benchmarks.qualification.single_kernel_microbench."
                "runtime_memory_observation", return_value=observation), patch(
                    "benchmarks.qualification.single_kernel_microbench."
                    "working_set_bytes", side_effect=[100, 130]), patch(
                        "benchmarks.qualification.single_kernel_microbench."
                        "process_gpu_memory_mib", side_effect=[10.0, 25.0]):
            with self.assertRaises(_StabilityReplayError) as captured:
                _run_stability(
                    FakeTi(), launch, 10, 10, True, "vanilla")
        evidence = captured.exception.evidence
        self.assertEqual(evidence["completed_replays"], 2)
        self.assertEqual(evidence["failed_replay_one_based"], 3)
        self.assertEqual(evidence["rss_delta_bytes"], 30)
        self.assertEqual(evidence["gpu_delta_mib"], 15.0)

    def test_native_reduce_route_requires_the_declared_cuda_plan(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_reduce_ndarray"

        class Workspace:
            _native_reduce_plan = Plan()
            workspace_bytes_current = 4096
            workspace_bytes_peak = 4096

        passed = _native_reduce_route(
            Workspace(), "forge", "cuda", "a" * 64)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["observed_method"],
                         "cuda_device_reduce_ndarray")
        Plan.method_name = "unexpected_fallback"
        failed = _native_reduce_route(
            Workspace(), "forge", "cuda", "a" * 64)
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
            _native_transform_route(
                Workspace(), "forge", "cuda", "a" * 64)["passed"])
        Plan.backend = "field_kernel"
        self.assertFalse(
            _native_transform_route(
                Workspace(), "forge", "cuda", "a" * 64)["passed"])

    def test_native_gather_route_requires_cached_native_plan(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_gather_ndarray"

        class Workspace:
            _native_indexed_copy_plan = Plan()
            workspace_bytes_current = 0
            workspace_bytes_peak = 0

        route = _native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=False,
            kernel_source_sha256="a" * 64)
        self.assertTrue(route["passed"])
        self.assertEqual(route["observed_method"],
                         "cuda_device_gather_ndarray")
        Plan.method_name = "gather_i32_ndarray_kernel_fallback"
        self.assertFalse(_native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=False,
            kernel_source_sha256="a" * 64)["passed"])

    def test_native_gather_kernel_route_proves_benchmark_control(self):
        source_sha256 = "a" * 64
        route = _native_indexed_copy_route(
            None, "forge_kernel", "cuda", scatter=False,
            kernel_source_sha256=source_sha256)
        self.assertTrue(route["passed"])
        self.assertEqual(route["adapter"], "benchmark_defined_ti_kernel")
        self.assertEqual(route["kernel_source_owner"], "benchmark")
        self.assertEqual(route["kernel_source_sha256"], source_sha256)
        self.assertFalse(route["helper_api_used"])
        self.assertFalse(route["workspace_present"])
        self.assertEqual(route["ti_kernel_invocations_per_replay"], 1)
        self.assertFalse(route["physical_backend_launches_assumed"])
        self.assertFalse(_native_indexed_copy_route(
            object(), "forge_kernel", "cuda", scatter=False,
            kernel_source_sha256=source_sha256)["passed"])
        self.assertFalse(_native_indexed_copy_route(
            None, "forge_kernel", "cuda", scatter=False,
            kernel_source_sha256="short")["passed"])

    def test_native_scatter_route_distinguishes_gather_plan(self):
        class Plan:
            backend = "cuda_device"
            method_name = "cuda_device_scatter_ndarray"

        class Workspace:
            _native_indexed_copy_plan = Plan()
            workspace_bytes_current = 0
            workspace_bytes_peak = 0

        self.assertTrue(_native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=True,
            kernel_source_sha256="a" * 64)["passed"])
        self.assertFalse(_native_indexed_copy_route(
            Workspace(), "forge", "cuda", scatter=False,
            kernel_source_sha256="a" * 64)["passed"])

    def test_native_scatter_kernel_route_proves_benchmark_control(self):
        source_sha256 = "b" * 64
        route = _native_indexed_copy_route(
            None, "vanilla_kernel", "cuda", scatter=True,
            kernel_source_sha256=source_sha256)
        self.assertTrue(route["passed"])
        self.assertEqual(route["adapter"], "benchmark_defined_ti_kernel")
        self.assertEqual(route["kernel_source_owner"], "benchmark")
        self.assertEqual(route["kernel_source_sha256"], source_sha256)
        self.assertFalse(route["helper_api_used"])
        self.assertFalse(route["workspace_present"])
        self.assertEqual(route["ti_kernel_invocations_per_replay"], 1)
        self.assertFalse(route["physical_backend_launches_assumed"])
        self.assertFalse(_native_indexed_copy_route(
            object(), "vanilla_kernel", "cuda", scatter=True,
            kernel_source_sha256=source_sha256)["passed"])
        self.assertFalse(_native_indexed_copy_route(
            None, "vanilla_kernel", "cuda", scatter=True,
            kernel_source_sha256="short")["passed"])

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
