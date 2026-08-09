import argparse
import math
import os
import unittest

from benchmarks.qualification.single_kernel_microbench import (
    _ExclusiveBenchmarkLock,
    QUALIFICATION_MINIMUMS,
    balanced_pair_orders,
    paired_log_summary,
    qualification_policy_errors,
    select_common_batch,
)


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


if __name__ == "__main__":
    unittest.main()
