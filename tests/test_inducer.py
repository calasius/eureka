"""Validation of the inducer on Levels 1-2 (build order step 3).

The inducer must:
  * recover a rule behaviorally EQUIVALENT to the planted one on Levels 1-2,
  * keep it simple (not bury the signal in spurious clauses),
  * report NOTHING on the negative control (no holdout-compressing rule),
  * recombine a promoted library primitive (Mechanism 2),
  * never inspect test/ground-truth to build or pick a rule (honesty boundary).
"""

import os
import tempfile
import unittest

from rule_induction.dataset import generate_level, split_train_test
from rule_induction.inducer import discover, induce, mine_predicates, render_rule
from rule_induction.librarian import Librarian
from rule_induction.metrics import recovered


def _level(level, n=900, seed=0, ratio=0.7):
    cases, gt = generate_level(level, n=n, seed=seed)
    train, test = split_train_test(cases, ratio, seed=seed)
    return train, test, gt


class RecoversRule(unittest.TestCase):
    def test_level1_recovers_general_successor_rule(self):
        train, test, gt = _level("level1")
        result = discover(train, test)
        self.assertTrue(result["found_rule"])
        best = result["best"]["hypothesis"]
        self.assertTrue(recovered("level1", gt, best)["equivalent"],
                        msg=f"got: {best['description']}")
        self.assertGreater(result["best"]["verdict"]["bits_saved"], 8.0)

    def test_level2_recovers_under_noise(self):
        train, test, gt = _level("level2")
        result = discover(train, test)
        self.assertTrue(result["found_rule"])
        best = result["best"]["hypothesis"]
        self.assertTrue(recovered("level2", gt, best)["equivalent"],
                        msg=f"got: {best['description']}")

    def test_level0_recovers_marker(self):
        train, test, gt = _level("level0")
        result = discover(train, test)
        self.assertTrue(result["found_rule"])
        self.assertTrue(recovered("level0", gt, result["best"]["hypothesis"])["equivalent"])


class StaysSimple(unittest.TestCase):
    def test_winner_is_not_overfit(self):
        # The level-1 winner should be compact (a handful of clauses), not a
        # memorizing thicket — the holdout MDL prunes spurious clauses.
        train, test, _gt = _level("level1")
        best = discover(train, test)["best"]["hypothesis"]
        self.assertLessEqual(len(best["params"]["clauses"]), 4)


class NoHallucination(unittest.TestCase):
    def test_negative_control_yields_no_rule(self):
        for seed in range(3):
            train, test, _gt = _level("neg", seed=seed)
            result = discover(train, test)
            self.assertFalse(result["found_rule"],
                             msg=f"hallucinated a rule on NEG seed {seed}: "
                                 f"{result['best'] and result['best']['hypothesis']['description']}")


class Recombination(unittest.TestCase):
    def test_library_primitive_enters_the_pool(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Librarian(os.path.join(d, "library"))
            lib.promote({
                "name": "successor T1->F1", "kind": "rule",
                "description": "promoted primitive",
                "rule_id": "R1_typed_successor",
                "spec": {"pairs": [["T1", "F1"]], "lag": 2},
                "mdl": {"bits_saved": 30.0, "compresses": "level1"},
                "provenance": {"level_origin": "level1"},
            })
            train, _test, _gt = _level("level2")
            pool = mine_predicates(train, library=lib)
            # The promoted pair is available as a ready-made predicate.
            self.assertTrue(any(
                s["pred"] == "typed_successor" and s["params"]["pairs"] == [["T1", "F1"]]
                for s in pool))


class HonestyBoundary(unittest.TestCase):
    def test_induce_uses_only_train(self):
        # induce() takes train alone; corrupting test must not change the proposals.
        train, test, _gt = _level("level1")
        c1 = [h["description"] for h in induce(train)]
        for c in test:
            c["outcome"] = "approve"            # poison the holdout labels
        c2 = [h["description"] for h in induce(train)]
        self.assertEqual(c1, c2)                # proposals are a function of train only


class Render(unittest.TestCase):
    def test_render_is_legible(self):
        params = {"clauses": [{"pred": {"pred": "marker", "params": {"marker": "evt_X"}},
                               "outcome": "reject"}], "default": "approve"}
        self.assertEqual(render_rule(params), "IF has(evt_X) THEN reject ELSE approve")

    def test_render_conjunction(self):
        params = {"clauses": [{"all": [{"pred": "marker", "params": {"marker": "T1"}},
                                       {"pred": "marker", "params": {"marker": "evt_Y"}}],
                               "outcome": "reject"}], "default": "approve"}
        self.assertEqual(render_rule(params),
                         "IF has(T1) AND has(evt_Y) THEN reject ELSE approve")


class Conjunctions(unittest.TestCase):
    def test_conjunctive_clause_executes(self):
        from rule_induction.rules import make_labeler
        params = {"clauses": [{"all": [{"pred": "marker", "params": {"marker": "a"}},
                                       {"pred": "marker", "params": {"marker": "b"}}],
                               "outcome": "reject"}], "default": "approve"}
        f = make_labeler("DL_decision_list", params)
        ev = lambda *ts: [{"t": i, "type": t, "attrs": {}} for i, t in enumerate(ts)]
        self.assertEqual(f(ev("a", "b")), "reject")   # both -> fires
        self.assertEqual(f(ev("a")), "approve")        # only one -> default
        self.assertEqual(f(ev("b")), "approve")

    def test_level4_finds_partial_structure(self):
        # Level 4 is expected to be irregular; the inducer should still find a
        # compressing rule with materially-better-than-chance accuracy.
        train, test, _gt = _level("level4")
        result = discover(train, test)
        self.assertTrue(result["found_rule"])
        self.assertGreater(result["best"]["verdict"]["test_accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
