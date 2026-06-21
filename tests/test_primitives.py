"""Tests for invented primitives + two-part MDL (the abstraction-invention loop)."""

import tempfile
import unittest

from rule_induction import arbiter, primitives
from rule_induction.dataset import split_train_test
from rule_induction.demo_invent import _composed, _gen
from rule_induction.librarian import Librarian


class TestInvention(unittest.TestCase):
    def test_compile_produces_runnable_predict(self):
        src = primitives.compile_composed(_composed("up", "down", "x"))
        self.assertIn("def count_gt", src)
        self.assertIn("def predict(events):", src)

    def test_reuse_is_cheaper_than_invention(self):
        hyp = _composed("up", "down", "x")
        full = primitives.program_bits_composed(hyp, frozenset())
        named = primitives.program_bits_composed(hyp, frozenset({"count_gt"}))
        self.assertLess(named, full)   # a known primitive costs a pointer, not its body

    def test_invent_then_promote_then_reuse(self):
        lib = Librarian(tempfile.mkdtemp(prefix="eureka_test_"), use_git=False)

        a_tr, a_te = split_train_test(_gen(0, 800, "up", "down"), 0.6, 0)
        vA = arbiter.adjudicate(_composed("up", "down", "x"), a_tr, a_te, librarian=lib,
                                known_primitives=lib.known_primitive_names(), level_origin="A")
        self.assertEqual(vA["decision"], "accept")
        self.assertTrue(vA["promoted"])
        self.assertIn("count_gt", lib.known_primitive_names())   # now shared vocabulary

        b_tr, b_te = split_train_test(_gen(1, 800, "win", "loss"), 0.6, 1)
        vB = arbiter.evaluate(_composed("win", "loss", "x"), b_tr, b_te,
                              known_primitives=lib.known_primitive_names())
        self.assertEqual(vB["test_accuracy"], 1.0)
        self.assertLess(vB["program_bits"], vA["program_bits"])   # amortized by reuse


if __name__ == "__main__":
    unittest.main()
