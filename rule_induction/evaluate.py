"""Evaluation harness — the Section 7 metrics, with variance (Section 6).

Runs the full inducer -> arbiter loop across levels x seeds and reports the
*distribution*, never the best case. Because the planted rule is known, it can
measure things impossible on real data:

* **Rule recovery** — did the inducer produce a rule *logically equivalent* to the
  planted one (behavioral equivalence on a fresh sample)?
* **Hallucination** — on the negative control, any accepted rule is a false
  positive; on positive levels, an accepted rule that is *not* equivalent.
* **Sample efficiency** — the smallest training size that recovers the rule
  (opt-in; good inducers need few cases).

These are offline diagnostics computed against ground truth; they never feed the
arbiter (that would leak the answer). The arbiter still judges only on holdout.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List, Optional

from .dataset import generate_level, split_train_test
from .inducer import discover
from .librarian import Librarian
from .metrics import recovered
from .rules import NO_RULE

ALL = ["level0", "level1", "level2", "level3", "level4", "level5", "neg"]


def _mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": statistics.mean(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "n": len(xs)}


def evaluate_level(level: str, *, seeds: int = 5, n: int = 600, ratio: float = 0.7,
                   library: Optional[Librarian] = None) -> Dict[str, Any]:
    """Run the loop over many seeds for one level; aggregate the metrics."""
    per_seed = []
    for seed in range(seeds):
        cases, gt = generate_level(level, n=n, seed=seed)
        train, test = split_train_test(cases, ratio, seed=seed)
        result = discover(train, test, library=library)
        is_neg = gt["planted_rule_id"] == NO_RULE
        row: Dict[str, Any] = {"seed": seed, "found": result["found_rule"]}
        if result["found_rule"]:
            v = result["best"]["verdict"]
            row["test_accuracy"] = v["test_accuracy"]
            row["bits_saved"] = v["bits_saved"]
            row["n_clauses"] = len(result["best"]["hypothesis"]["params"]["clauses"])
            row["equivalent"] = (False if is_neg
                                 else recovered(level, gt, result["best"]["hypothesis"])["equivalent"])
        per_seed.append(row)

    found = [r for r in per_seed if r["found"]]
    equiv = [r for r in found if r.get("equivalent")]
    is_neg = level == "neg"
    # Hallucination = false positive from noise: only meaningful on the negative
    # control. On positive levels a non-equivalent-but-accurate rule is partial
    # recovery (see test_accuracy), not a fabrication — recovery_rate tells that.
    return {
        "level": level,
        "seeds": seeds,
        "found_rate": len(found) / seeds,
        "recovery_rate": (None if is_neg else len(equiv) / seeds),
        "hallucination_rate": (len(found) / seeds if is_neg else None),
        "test_accuracy": _mean_std([r["test_accuracy"] for r in found]),
        "bits_saved": _mean_std([r["bits_saved"] for r in found]),
        "clauses": _mean_std([r["n_clauses"] for r in found]),
        "per_seed": per_seed,
    }


def sample_efficiency(level: str, *, candidate_ns: Optional[List[int]] = None,
                      seeds: int = 5, ratio: float = 0.7) -> Optional[int]:
    """Smallest training size recovering the rule in a majority of seeds (or None)."""
    if level == "neg":
        return None
    candidate_ns = candidate_ns or [60, 120, 240, 480, 960]
    need = (seeds // 2) + 1
    for n in candidate_ns:
        hits = 0
        for seed in range(seeds):
            cases, gt = generate_level(level, n=n, seed=seed)
            train, test = split_train_test(cases, ratio, seed=seed)
            result = discover(train, test)
            if result["found_rule"] and recovered(level, gt, result["best"]["hypothesis"])["equivalent"]:
                hits += 1
        if hits >= need:
            return n
    return None


def run_benchmark(levels: Optional[List[str]] = None, *, seeds: int = 5, n: int = 600,
                  with_sample_efficiency: bool = False) -> Dict[str, Any]:
    levels = levels or ALL
    reports = [evaluate_level(lv, seeds=seeds, n=n) for lv in levels]
    if with_sample_efficiency:
        for rep in reports:
            rep["min_n_to_recover"] = sample_efficiency(rep["level"], seeds=seeds)
    return {"seeds": seeds, "n": n, "levels": reports}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _fmt(ms: Dict[str, float], pct: bool = False) -> str:
    if ms["n"] == 0:
        return "    —    "
    scale, suffix = (100, "%") if pct else (1, "")
    return f"{ms['mean'] * scale:5.1f}{suffix}±{ms['std'] * scale:.0f}"


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark the rule-induction system across levels x seeds.")
    p.add_argument("--levels", nargs="+", default=None, choices=ALL, metavar="LEVEL")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("-n", type=int, default=600, help="cases per (level, seed) [default 600]")
    p.add_argument("--sample-efficiency", action="store_true",
                   help="also report the smallest n that recovers each rule")
    args = p.parse_args(argv)

    report = run_benchmark(args.levels, seeds=args.seeds, n=args.n,
                           with_sample_efficiency=args.sample_efficiency)

    print(f"Rule-induction benchmark — {report['seeds']} seeds, n={report['n']} per level")
    print("(distribution across seeds; recovery = behaviorally equivalent to the planted rule)\n")
    head = f"{'level':8} {'recovery':>9} {'halluc.':>8} {'found':>6} {'test_acc':>11} {'bits_saved':>13} {'clauses':>8}"
    if args.sample_efficiency:
        head += f" {'min_n':>6}"
    print(head)
    print("-" * len(head))
    for rep in report["levels"]:
        rec = "   —   " if rep["recovery_rate"] is None else f"{rep['recovery_rate']*100:5.0f}%  "
        hal = "   —  " if rep["hallucination_rate"] is None else f"{rep['hallucination_rate']*100:5.0f}%"
        line = (f"{rep['level']:8} {rec:>9} {hal:>7} "
                f"{rep['found_rate']*100:5.0f}% {_fmt(rep['test_accuracy'], pct=True):>11} "
                f"{_fmt(rep['bits_saved']):>13} {_fmt(rep['clauses']):>8}")
        if args.sample_efficiency:
            mn = rep.get("min_n_to_recover")
            line += f" {('>'+str(960) if mn is None else mn):>6}"
        print(line)

    print("\nReading: levels 0–3 should recover solidly; level 4 partial; level 5 fails")
    print("(the honest analogy limit); neg must recover NOTHING (hallucination ~0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
