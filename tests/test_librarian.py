"""Tests for the Librarian persistent store."""

import os
import tempfile
import unittest

from rule_induction.librarian import Librarian, PromotionRejected


def _entry(bits, name="typed-successor t1->f1", **over):
    e = {
        "name": name,
        "kind": "rule",
        "description": "Tk followed by Fk within lag -> reject.",
        "rule_id": "R1_typed_successor",
        "spec": {"pairs": [["T1", "F1"]], "lag": 2},
        "mdl": {"bits_saved": bits, "compresses": "level1 outcomes"},
        "provenance": {"run_id": "run_1", "investigator": "simplicity", "level_origin": "level1"},
    }
    e.update(over)
    return e


class PromotionThreshold(unittest.TestCase):
    def test_rejects_below_library_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Librarian(os.path.join(d, "library"))
            self.assertEqual(lib.library_threshold(), 8.0)
            with self.assertRaises(PromotionRejected):
                lib.promote(_entry(bits=3.0))   # below 8.0
            self.assertEqual(lib.list(), [])

    def test_accepts_above_threshold_and_persists(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "library")
            lib = Librarian(root)
            rec = lib.promote(_entry(bits=42.0))
            self.assertEqual(rec["status"], "promoted")
            self.assertEqual(rec["version"], 1)
            # File on disk + index updated.
            self.assertTrue(os.path.exists(os.path.join(root, "entries", rec["id"] + ".json")))
            self.assertEqual(len(lib.list()), 1)
            # A fresh handle reads the same persisted state.
            lib2 = Librarian(root)
            self.assertEqual(lib2.get(rec["id"])["mdl"]["bits_saved"], 42.0)


class Versioning(unittest.TestCase):
    def test_repromote_only_on_improvement(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Librarian(os.path.join(d, "library"))
            r1 = lib.promote(_entry(bits=20.0))
            with self.assertRaises(PromotionRejected):
                lib.promote(_entry(bits=20.0))           # not an improvement
            r2 = lib.promote(_entry(bits=25.0))          # strictly better -> v2
            self.assertEqual(r1["version"], 1)
            self.assertEqual(r2["version"], 2)
            self.assertEqual(len(lib.list()), 1)         # still one logical entry


class Validation(unittest.TestCase):
    def test_missing_fields_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Librarian(os.path.join(d, "library"))
            with self.assertRaises(ValueError):
                lib.promote({"name": "x", "kind": "rule"})           # no description/mdl
            with self.assertRaises(ValueError):
                lib.promote(_entry(bits=10.0, kind="banana"))        # bad kind


class RevertAndAudit(unittest.TestCase):
    def test_revert_removes_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "library")
            lib = Librarian(root)
            rec = lib.promote(_entry(bits=30.0))
            out = lib.revert(rec["id"], reason="false positive on NEG")
            self.assertTrue(out["reverted"])
            self.assertEqual(lib.list(), [])
            self.assertFalse(os.path.exists(os.path.join(root, "entries", rec["id"] + ".json")))

    def test_git_history_records_promote_and_revert(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Librarian(os.path.join(d, "library"))
            if not lib.use_git:
                self.skipTest("git unavailable")
            rec = lib.promote(_entry(bits=30.0))
            lib.revert(rec["id"], reason="cleanup")
            log = lib.log()
            self.assertIn("promote", log)
            self.assertIn("revert", log)


class WorksWithoutGit(unittest.TestCase):
    def test_no_git_still_persists(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Librarian(os.path.join(d, "library"), use_git=False)
            rec = lib.promote(_entry(bits=15.0))
            self.assertIsNone(rec.get("commit"))
            self.assertEqual(len(lib.list()), 1)


if __name__ == "__main__":
    unittest.main()
