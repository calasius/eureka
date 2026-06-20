"""Tests for the synthetic data generator (Skill 3).

The most important invariant: the recorded ground truth labeler must reproduce
*every* outcome in the generated cases. If it does, ground truth is exact and the
scorer can trust it. We also check label balance (no degenerate single-class
levels) and that the negative control is genuinely unpredictable.
"""

import unittest

from rule_induction.dataset import build, generate_level, split_train_test
from rule_induction.levels import ALL_LEVELS
from rule_induction.rules import NO_RULE, make_labeler

POSITIVE_LEVELS = [lv for lv in ALL_LEVELS if lv != "neg"]


class GroundTruthExactness(unittest.TestCase):
    def test_labeler_reproduces_every_outcome(self):
        for level in POSITIVE_LEVELS:
            for seed in range(3):
                cases, gt = generate_level(level, n=150, seed=seed)
                labeler = make_labeler(gt["planted_rule_id"], gt["params"])
                for c in cases:
                    self.assertEqual(
                        labeler(c["events"]), c["outcome"],
                        f"{level} seed {seed} case {c['case_id']}: labeler disagrees",
                    )

    def test_levels_are_not_degenerate(self):
        # Each positive level must exhibit at least two distinct outcomes.
        for level in POSITIVE_LEVELS:
            cases, gt = generate_level(level, n=200, seed=0)
            outcomes = {c["outcome"] for c in cases}
            self.assertGreaterEqual(
                len(outcomes), 2, f"{level} produced only {outcomes}")
            # And the minority class is non-trivial (>5%).
            dist = gt["label_distribution"]
            self.assertGreater(min(dist.values()) / sum(dist.values()), 0.05,
                               f"{level} is too imbalanced: {dist}")


class CaseFileFormat(unittest.TestCase):
    def test_required_keys_and_no_truth_leak(self):
        cases, gt = generate_level("level1", n=20, seed=1)
        for c in cases:
            for key in ("case_id", "events", "outcome", "level"):
                self.assertIn(key, c)
            # Generator output carries no planted_rule_id (scorer-only).
            self.assertNotIn("planted_rule_id", c)
            self.assertTrue(all({"t", "type", "attrs"} <= set(e) for e in c["events"]))
        self.assertEqual(gt["planted_rule_id"], "R1_typed_successor")


class SuccessorIsNotMerePresence(unittest.TestCase):
    """Level 1 must distinguish 'A then B within lag' from 'A and B present'."""

    def test_hard_negatives_exist(self):
        cases, _ = generate_level("level1", n=400, seed=2)
        # Some approve cases contain BOTH a trigger and a confirm (too-far / wrong
        # order), proving presence alone does not determine the outcome.
        def has(c, t):
            return any(e["type"] == t for e in c["events"])
        tricky = [c for c in cases
                  if c["outcome"] == "approve" and has(c, "T1") and has(c, "F1")]
        self.assertTrue(tricky, "no hard negatives: level1 reduces to mere presence")


class NegativeControl(unittest.TestCase):
    def test_no_labeler(self):
        _, gt = generate_level("neg", n=50, seed=0)
        self.assertEqual(gt["planted_rule_id"], NO_RULE)
        with self.assertRaises(ValueError):
            make_labeler(NO_RULE, gt["params"])

    def test_marker_does_not_predict_outcome(self):
        # The bait event evt_X must not correlate strongly with the outcome.
        cases, _ = generate_level("neg", n=1000, seed=0)
        with_x = [c for c in cases if any(e["type"] == "evt_X" for e in c["events"])]
        self.assertTrue(with_x)
        rate = sum(c["outcome"] == "reject" for c in with_x) / len(with_x)
        self.assertTrue(0.35 < rate < 0.65, f"evt_X predicts outcome (reject rate {rate:.2f})")


class Holdout(unittest.TestCase):
    def test_split_is_deterministic_and_partitions(self):
        cases, _ = generate_level("level2", n=100, seed=0)
        tr1, te1 = split_train_test(cases, 0.7, seed=0)
        tr2, te2 = split_train_test(cases, 0.7, seed=0)
        self.assertEqual([c["case_id"] for c in tr1], [c["case_id"] for c in tr2])
        self.assertEqual(len(tr1) + len(te1), len(cases))
        ids = {c["case_id"] for c in tr1} | {c["case_id"] for c in te1}
        self.assertEqual(len(ids), len(cases))  # disjoint, covering
        self.assertAlmostEqual(len(tr1) / len(cases), 0.7, delta=0.02)


class BuildOutput(unittest.TestCase):
    def test_build_writes_manifest_and_files(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            manifest = build(d, levels=["level0", "neg"], n=30, seeds=2, ratio=0.7)
            self.assertEqual(len(manifest["entries"]), 4)
            for e in manifest["entries"]:
                for fname in ("train.jsonl", "test.jsonl", "ground_truth.json"):
                    self.assertTrue(os.path.exists(os.path.join(e["dir"], fname)))
            self.assertTrue(os.path.exists(os.path.join(d, "manifest.json")))
            # train.jsonl lines are valid JSON case-files.
            p = os.path.join(manifest["entries"][0]["dir"], "train.jsonl")
            with open(p) as fh:
                rows = [json.loads(line) for line in fh]
            self.assertTrue(rows and all("outcome" in r for r in rows))


if __name__ == "__main__":
    unittest.main()
