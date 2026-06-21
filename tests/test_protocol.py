"""Tests for unsupervised protocol state-machine inference."""

import unittest

from rule_induction.protocol import (accepts, autoname, demo_traces, infer,
                                      to_markdown, to_mermaid)


class TestProtocolInference(unittest.TestCase):
    def setUp(self):
        self.sm, self.stats = infer(demo_traces(n=300, seed=0), k=2)
        autoname(self.sm)

    def test_recovers_compact_machine(self):
        # Handshake(3) + established + data(2) + teardown(3) collapses to 8 states.
        self.assertEqual(self.stats["dfa_states"], 8)
        self.assertLess(self.stats["dfa_states"], self.stats["pta_states"])

    def test_compresses_vs_structureless_baseline(self):
        self.assertGreater(self.stats["compression"]["bits_saved"], 0)

    def test_generalises_to_unseen_loop_counts(self):
        # Trained on 0..5 data rounds; must accept 10 (the loop generalised).
        novel = ["SYN", "SYN_ACK", "ACK"] + ["DATA", "DATA_ACK"] * 10 + ["FIN", "FIN_ACK", "ACK"]
        self.assertTrue(accepts(self.sm, novel))

    def test_rejects_malformed(self):
        self.assertFalse(accepts(self.sm, ["SYN", "ACK", "DATA"]))         # skips SYN_ACK
        self.assertFalse(accepts(self.sm, ["DATA", "DATA_ACK"]))           # no handshake

    def test_renders_mermaid_and_markdown(self):
        mer = to_mermaid(self.sm)
        self.assertIn("stateDiagram-v2", mer)
        self.assertIn("[*] -->", mer)
        md = to_markdown(self.sm, self.stats)
        self.assertIn("```mermaid", md)
        self.assertIn("Transition table", md)


if __name__ == "__main__":
    unittest.main()
