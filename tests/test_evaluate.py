"""Tests for the evaluation harness (Sections 6-7 metrics with variance)."""

import unittest

from rule_induction.evaluate import evaluate_level, run_benchmark, sample_efficiency


class LevelMetrics(unittest.TestCase):
    def test_level0_recovers_every_seed(self):
        rep = evaluate_level("level0", seeds=3, n=300)
        self.assertEqual(rep["recovery_rate"], 1.0)
        self.assertEqual(rep["found_rate"], 1.0)
        self.assertGreater(rep["test_accuracy"]["mean"], 0.99)
        self.assertEqual(len(rep["per_seed"]), 3)

    def test_negative_control_no_hallucination(self):
        rep = evaluate_level("neg", seeds=3, n=400)
        self.assertEqual(rep["found_rate"], 0.0)
        self.assertEqual(rep["hallucination_rate"], 0.0)
        self.assertIsNone(rep["recovery_rate"])


class Benchmark(unittest.TestCase):
    def test_report_shape(self):
        report = run_benchmark(["level0", "neg"], seeds=2, n=200)
        self.assertEqual(len(report["levels"]), 2)
        for rep in report["levels"]:
            for key in ("recovery_rate", "hallucination_rate", "found_rate",
                        "test_accuracy", "bits_saved", "clauses"):
                self.assertIn(key, rep)


class SampleEfficiency(unittest.TestCase):
    def test_level0_needs_few_cases(self):
        # The simplest level should recover from a small training set.
        min_n = sample_efficiency("level0", candidate_ns=[60, 120, 240], seeds=3)
        self.assertIsNotNone(min_n)
        self.assertLessEqual(min_n, 240)

    def test_neg_has_no_sample_efficiency(self):
        self.assertIsNone(sample_efficiency("neg"))


if __name__ == "__main__":
    unittest.main()
