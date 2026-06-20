"""Validation of the arbiter on Levels 0-1 (build order step 2).

The arbiter must:
  * ACCEPT the true planted rule (large positive bits_saved, ~perfect holdout),
  * REJECT a memorizer (huge program + holdout failure -> negative bits_saved),
  * RESIST the NEG bait (no rule compresses rule-free data),
  * run code-hypotheses in a sandbox and reject dirty ones,
  * promote an accepted rule into the Librarian.
"""

import json
import os
import tempfile
import unittest

from rule_induction import sandbox
from rule_induction.arbiter import adjudicate, evaluate
from rule_induction.dataset import generate_level, split_train_test
from rule_induction.librarian import Librarian
from rule_induction.metrics import recovered


def _level(level, n=900, seed=0, ratio=0.7):
    cases, gt = generate_level(level, n=n, seed=seed)
    train, test = split_train_test(cases, ratio, seed=seed)
    return train, test, gt


def _true_rule_hypothesis(gt):
    return {"kind": "rule", "rule_id": gt["planted_rule_id"], "params": gt["params"],
            "name": gt["planted_rule_id"], "description": "ground-truth rule"}


class AcceptsTrueRule(unittest.TestCase):
    def test_level0(self):
        train, test, gt = _level("level0")
        v = evaluate(_true_rule_hypothesis(gt), train, test)
        self.assertEqual(v["decision"], "accept")
        self.assertGreater(v["bits_saved"], 8.0)          # clears even the library bar
        self.assertGreater(v["test_accuracy"], 0.99)

    def test_level1_recovers_general_rule(self):
        train, test, gt = _level("level1")
        hyp = _true_rule_hypothesis(gt)
        v = evaluate(hyp, train, test)
        self.assertEqual(v["decision"], "accept")
        self.assertGreater(v["bits_saved"], 8.0)
        # And it is behaviorally equivalent to the planted rule on a fresh sample.
        self.assertTrue(recovered("level1", gt, hyp)["equivalent"])


class RejectsMemorizer(unittest.TestCase):
    def test_memorizer_loses_to_short_rule(self):
        train, test, gt = _level("level1")
        # A lookup table over training signatures; unseen -> majority class.
        table = {"|".join(e["type"] for e in c["events"]): c["outcome"] for c in train}
        majority = max({c["outcome"] for c in train},
                       key=lambda o: sum(x["outcome"] == o for x in train))
        source = (
            "TABLE = " + json.dumps(table) + "\n"
            "def predict(events):\n"
            "    sig = '|'.join(e['type'] for e in events)\n"
            f"    return TABLE.get(sig, {majority!r})\n"
        )
        mem = {"kind": "code", "source": source, "name": "memorizer"}
        vm = evaluate(mem, train, test)
        vt = evaluate(_true_rule_hypothesis(gt), train, test)
        self.assertEqual(vm["decision"], "reject")
        self.assertLess(vm["bits_saved"], 0.0)
        # The short true rule beats the memorizer by a wide margin.
        self.assertGreater(vt["bits_saved"], vm["bits_saved"] + 100)
        self.assertGreater(vm["program_bits"], vt["program_bits"] * 10)


class ResistsNegBait(unittest.TestCase):
    def test_no_rule_compresses_negative_control(self):
        train, test, _gt = _level("neg")
        # The most tempting simple rule on the bait events:
        hyp = {"kind": "rule", "rule_id": "R0_marker_presence",
               "params": {"marker": "evt_X", "hit": "reject", "miss": "approve"}}
        v = evaluate(hyp, train, test)
        self.assertEqual(v["decision"], "reject")
        self.assertLessEqual(v["bits_saved"], 2.0)


class Sandbox(unittest.TestCase):
    def test_clean_code_matches_symbolic(self):
        train, test, gt = _level("level0")
        code = {"kind": "code",
                "source": "def predict(events):\n"
                          "    return 'reject' if any(e['type']=='evt_X' for e in events) else 'approve'"}
        v = evaluate(code, train, test)
        self.assertEqual(v["decision"], "accept")
        self.assertTrue(v["sandbox_clean"])
        self.assertGreater(v["test_accuracy"], 0.99)

    def test_raising_code_is_rejected(self):
        train, test, _ = _level("level0", n=100)
        bad = {"kind": "code", "source": "def predict(events):\n    raise RuntimeError('boom')"}
        v = evaluate(bad, train, test)
        self.assertEqual(v["decision"], "reject")
        self.assertFalse(v["sandbox_clean"])

    def test_timeout_is_caught(self):
        with self.assertRaises(sandbox.SandboxError):
            sandbox.run_code("def predict(events):\n    \n    x=0\n    while True:\n        x+=1",
                             [{"events": []}], timeout_s=1.0)

    def test_filesystem_write_blocked(self):
        # RLIMIT_FSIZE=0 + no 'open' in builtins -> any disk write fails.
        src = "def predict(events):\n    open('/tmp/evil','w').write('x')\n    return 'approve'"
        with self.assertRaises(sandbox.SandboxError):
            sandbox.run_code(src, [{"events": []}], timeout_s=3.0)


class PromotionWiring(unittest.TestCase):
    def test_accepted_rule_is_promoted(self):
        train, test, gt = _level("level1")
        with tempfile.TemporaryDirectory() as d:
            lib = Librarian(os.path.join(d, "library"))
            v = adjudicate(_true_rule_hypothesis(gt), train, test,
                           librarian=lib, level_origin="level1")
            self.assertEqual(v["decision"], "accept")
            self.assertTrue(v["promoted"])
            self.assertEqual(len(lib.list()), 1)
            stored = lib.list()[0]
            self.assertEqual(stored["rule_id"], gt["planted_rule_id"])
            self.assertGreaterEqual(stored["bits_saved"], lib.library_threshold())


if __name__ == "__main__":
    unittest.main()
