"""Rule-Induction Agent — core library.

This package currently ships the two pieces requested first in the build order
(Section 8 of rule_induction_agent_plan.md):

* ``rule_induction.dataset`` / ``rule_induction.levels`` / ``rule_induction.rules``
  — Skill 3, the synthetic data generator (the test bench).
* ``rule_induction.librarian`` — the persistent store ("the Librarian"), a
  git-versioned directory of accepted abstractions. NOT a skill; it is *state*
  written by the arbiter (on promotion) and read by the inducer (at generation).

The arbiter (Skill 2) and inducer (Skill 1) are intentionally not built yet.
"""

__version__ = "0.1.0"
