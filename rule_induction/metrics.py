"""Ground-truth metrics (Section 7) — what synthetic data uniquely enables.

These are *offline* checks, separate from the arbiter's holdout judgment: because
the planted rule is known, we can ask questions impossible on real data. They are
used to validate the system, NOT to make accept/reject decisions (that would leak
the answer to the arbiter).

* **Rule recovery (behavioral equivalence).** Two rules are "logically
  equivalent (even if written differently)" iff they agree on (almost) all
  outcomes over a fresh sample drawn from the same generator. We approximate
  logical equivalence with behavioral equivalence on held-out-from-everything
  data, which is the operational test.
* **Hallucination.** On the negative control there is no rule; any hypothesis the
  arbiter *accepts* there is a false positive.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .levels import GENERATORS
from .rules import NO_RULE, make_labeler


def _predict_rule(rule_id: str, params: Dict[str, Any], cases: List[Dict[str, Any]]) -> List[Any]:
    labeler = make_labeler(rule_id, params)
    return [labeler(c["events"]) for c in cases]


def behavioral_equivalence(level: str, ground_truth: Dict[str, Any],
                           hypothesis_predict, *, n: int = 1000, seed: int = 9999
                           ) -> Dict[str, Any]:
    """Agreement between a hypothesis and the planted rule on a FRESH sample.

    ``hypothesis_predict`` maps ``events -> outcome``. The sample seed is distinct
    from any training/holdout seed, so this measures generalisation, not memory.
    """
    cases, _gt = GENERATORS[level](n, seed)
    if ground_truth["planted_rule_id"] == NO_RULE:
        raise ValueError("the negative control has no rule to be equivalent to")
    planted = make_labeler(ground_truth["planted_rule_id"], ground_truth["params"])
    agree = sum(1 for c in cases
                if hypothesis_predict(c["events"]) == planted(c["events"]))
    rate = agree / len(cases)
    return {"agreement": rate, "n": len(cases), "equivalent": rate >= 0.99}


def recovered(level: str, ground_truth: Dict[str, Any], hypothesis: Dict[str, Any],
              *, n: int = 1000, seed: int = 9999) -> Dict[str, Any]:
    """Did the system recover a rule equivalent to the planted one? (rule hypotheses)"""
    if hypothesis.get("kind") != "rule":
        raise ValueError("recovered() expects a symbolic rule hypothesis")
    predict = make_labeler(hypothesis["rule_id"], hypothesis.get("params", {}))
    return behavioral_equivalence(level, ground_truth, predict, n=n, seed=seed)
