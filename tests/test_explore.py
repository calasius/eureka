"""Tests for the train-only scratchpad — runs analysis, and CANNOT touch the answer."""

import unittest

from rule_induction import sandbox

TRAIN = [
    {"case_id": "a", "events": [{"type": "x"}], "outcome": "yes"},
    {"case_id": "b", "events": [{"type": "y"}], "outcome": "no"},
    {"case_id": "c", "events": [{"type": "x"}], "outcome": "yes"},
]


class TestScratchpad(unittest.TestCase):
    def test_runs_and_returns_computation(self):
        src = ("def analyze(train):\n"
               "    return {'n': len(train),\n"
               "            'outcomes': sorted({c['outcome'] for c in train})}")
        r = sandbox.run_analysis(src, TRAIN)
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["outcomes"], ["no", "yes"])

    def test_file_access_is_blocked(self):
        # No `open` in the sandbox builtins -> cannot read test.jsonl / ground_truth.json.
        src = "def analyze(train):\n    return open('/etc/hostname').read()"
        with self.assertRaises(sandbox.SandboxError):
            sandbox.run_analysis(src, TRAIN)

    def test_import_is_blocked(self):
        src = "def analyze(train):\n    import os\n    return os.listdir('.')"
        with self.assertRaises(sandbox.SandboxError):
            sandbox.run_analysis(src, TRAIN)


if __name__ == "__main__":
    unittest.main()
