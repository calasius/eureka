"""Tests for the hard grammar dataset + cross-vocabulary transfer (parametric reuse)."""

import tempfile
import unittest

from rule_induction import arbiter
from rule_induction.arbiter import load_dataset
from rule_induction.dataset import split_train_test
from rule_induction.demo_grammar_transfer import PARENS, TAGS, hypothesis, make_generator
from rule_induction.grammar_hard import reference_hypothesis, rule_accept, write_dataset
from rule_induction.librarian import Librarian


class TestHardGrammar(unittest.TestCase):
    def test_reference_recovers_on_holdout(self):
        out = tempfile.mkdtemp(prefix="eureka_gram_")
        write_dataset(out, n=3000, seed=0, ratio=0.6)
        train, test, _ = load_dataset(out)
        v = arbiter.evaluate(reference_hypothesis(), train, test)
        self.assertEqual(v["decision"], "accept")
        self.assertEqual(v["test_accuracy"], 1.0)

    def test_partial_hypothesis_is_not_perfect(self):
        # nesting-only must miss the depth/count negatives -> a strictly harder rule.
        gen = make_generator(*PARENS, depth=3)
        cases = gen(1500, 3)
        acc = sum(c["outcome"] == "accept" for c in cases)
        self.assertTrue(0 < acc < len(cases))           # both classes present
        self.assertTrue(all(rule_accept(c["events"]) == (c["outcome"] == "accept") for c in cases))


class TestTransfer(unittest.TestCase):
    def test_invent_then_reuse_cross_alphabet(self):
        lib = Librarian(tempfile.mkdtemp(prefix="eureka_tr_"), use_git=False)

        # An expensive pushdown recogniser only clears MDL with enough holdout
        # evidence, so the dataset must be large enough for A to promote it.
        gA = make_generator(*PARENS, depth=3)
        trA, teA = split_train_test(gA(4000, 0), 0.6, 0)
        vA = arbiter.adjudicate(hypothesis({"kx": "kz", "wp": "wq"}), trA, teA, librarian=lib,
                                known_primitives=lib.known_primitive_names(), level_origin="A")
        self.assertEqual(vA["test_accuracy"], 1.0)
        self.assertTrue(vA["promoted"])

        gB = make_generator(*TAGS, depth=3)
        trB, teB = split_train_test(gB(4000, 1), 0.6, 1)
        vB = arbiter.adjudicate(hypothesis({"div_o": "div_c", "span_o": "span_c"}), trB, teB, librarian=lib,
                                known_primitives=lib.known_primitive_names(), level_origin="B")
        self.assertEqual(vB["test_accuracy"], 1.0)
        self.assertLess(vB["program_bits"], vA["program_bits"] / 4)   # reuse is far cheaper


if __name__ == "__main__":
    unittest.main()
