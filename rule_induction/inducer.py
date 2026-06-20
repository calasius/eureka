"""The Inducer — Skill 1, the orchestrator (single-investigator, Mechanisms 1-3).

Pipeline: encode -> base-mine a predicate pool -> generate candidate rules
(simplicity-ordered, residual-directed, recombining library primitives) -> hand
them to the **arbiter** -> keep the candidate that compresses the holdout best.

The honesty boundary: the inducer reads **train only** to *propose* hypotheses
(it is allowed to fit train). The **arbiter** then scores each candidate on the
**holdout** with the program length included, and *selects* — so a candidate that
overfits train but does not generalise loses to a shorter one. The inducer never
inspects test outcomes to build or pick a rule.

The three generative mechanisms realised here:
  * **Simplicity prior (M1).** Candidates are emitted as a nested sequence of
    increasing length (0 clauses, 1, 2, ...). Because the arbiter penalises
    program length on holdout, the simplest adequate rule wins — climbing in
    complexity only pays when the data demands it.
  * **Recombination (M2).** The predicate pool is seeded with primitives exploded
    from the library's promoted rules, so a discovery is reused, not re-derived.
  * **Surprise-directed search (M3).** Each new clause is mined from the cases the
    current rule still gets wrong (the residuals / uncovered set), not from where
    it already predicts well.

The hypothesis class is a legible **decision list** (`DL_decision_list`): the
recovered rule is an auditable artifact, which is the system's deliverable.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import arbiter as arb
from .librarian import Librarian
from .rules import _build_predicate

Case = Dict[str, Any]
PredSpec = Dict[str, Any]

MAX_LAG = 3
DEFAULT_MAX_CLAUSES = 6
MAX_CONJ = 3          # max predicates AND-ed in one clause (enables hierarchies)
MAX_POOL = 300
STRUCTURAL = {"open", "close"}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _majority(labels: List[Any]) -> Any:
    counts = Counter(labels)
    top = max(counts.values())
    return sorted([k for k, v in counts.items() if v == top], key=str)[0]


def _majority_count(labels: List[Any]) -> int:
    return max(Counter(labels).values()) if labels else 0


def _argmax_label(counts: Counter) -> Any:
    """Highest-count label, ties broken in sorted order (matches _majority)."""
    best = None
    for label in sorted(counts, key=str):
        if best is None or counts[label] > counts[best]:
            best = label
    return best


def _pred_complexity(spec: PredSpec) -> float:
    base = {"marker": 1.0, "attr_eq": 1.0, "count_at_least": 2.0, "typed_successor": 2.0}
    return base.get(spec["pred"], 3.0) + 0.1 * float(spec.get("params", {}).get("lag", 0))


# --------------------------------------------------------------------------- #
# Base mining — the predicate pool (the lego set)                              #
# --------------------------------------------------------------------------- #
def mine_predicates(train: List[Case], library: Optional[Librarian] = None) -> List[PredSpec]:
    """Discover candidate predicates actually supported by the train traces."""
    n = len(train)
    min_count = max(2, n // 50)
    type_counts: Counter = Counter()
    pair_support: Counter = Counter()
    pair_outcome: Dict[Tuple[str, str], Counter] = {}
    outcome_counts: Counter = Counter()
    attr_values: Dict[str, set] = {}

    for c in train:
        seq = [e["type"] for e in c["events"]]
        outcome = c["outcome"]
        outcome_counts[outcome] += 1
        for t in seq:
            if t not in STRUCTURAL:
                type_counts[t] += 1
        present: set = set()
        for i, a in enumerate(seq):
            if a in STRUCTURAL:
                continue
            for lag in range(1, MAX_LAG + 1):
                j = i + lag
                if j < len(seq) and seq[j] not in STRUCTURAL:
                    present.add((a, seq[j]))
        for pr in present:                       # count each pair once per case
            pair_support[pr] += 1
            pair_outcome.setdefault(pr, Counter())[outcome] += 1
        for e in c["events"]:
            if e["type"] == "open":
                for k, v in e.get("attrs", {}).items():
                    if isinstance(v, (str, bool, int)):
                        attr_values.setdefault(k, set()).add(v)

    pool: List[PredSpec] = []
    # marker(t)
    for t, _ in type_counts.most_common():
        pool.append({"pred": "marker", "params": {"marker": t}})
    # successor(a, b, lag): rank pairs by association with the outcome (excess
    # purity over the base rate), NOT by raw frequency — so a rare but perfectly
    # discriminative signal pair beats a common but uninformative noise pair.
    base_major_rate = (max(outcome_counts.values()) / n) if n else 0.0
    scored_pairs = []
    for pr, support in pair_support.items():
        if support < min_count:
            continue
        major_in_firing = pair_outcome[pr].most_common(1)[0][1]
        excess_purity = major_in_firing - support * base_major_rate
        scored_pairs.append((excess_purity, pr))
    scored_pairs.sort(key=lambda x: (-x[0], str(x[1])))
    for _, (a, b) in scored_pairs[:40]:
        for lag in range(1, MAX_LAG + 1):
            pool.append({"pred": "typed_successor", "params": {"pairs": [[a, b]], "lag": lag}})
    # count_at_least(t, n)
    for t, _ in type_counts.most_common(10):
        for k in (2, 3):
            pool.append({"pred": "count_at_least", "params": {"type": t, "n": k}})
    # attr_eq(key, value) — enables splitting multimodal populations
    for key, values in sorted(attr_values.items()):
        for v in sorted(values, key=str):
            pool.append({"pred": "attr_eq", "params": {"key": key, "value": v}})
    # Mechanism 2 — recombination: explode promoted library rules into predicates
    if library is not None:
        pool = _library_primitives(library) + pool

    # Dedup (stable) and cap by a simplicity-first ranking.
    seen, deduped = set(), []
    for spec in pool:
        key = json.dumps(spec, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(spec)
    deduped.sort(key=_pred_complexity)
    return deduped[:MAX_POOL]


def _library_primitives(library: Librarian) -> List[PredSpec]:
    """Turn promoted abstractions back into reusable predicates (Mechanism 2)."""
    out: List[PredSpec] = []
    for summary in library.list():
        entry = library.get(summary["id"])
        spec = entry.get("spec") or {}
        rid = entry.get("rule_id")
        if rid in ("R1_typed_successor", "R2_typed_successor_noisy") and "pairs" in spec:
            for pair in spec["pairs"]:
                out.append({"pred": "typed_successor",
                            "params": {"pairs": [pair], "lag": spec.get("lag", 2)}})
        elif rid == "R0_marker_presence" and "marker" in spec:
            out.append({"pred": "marker", "params": {"marker": spec["marker"]}})
        elif rid == "DL_decision_list":
            for clause in spec.get("clauses", []):
                out.extend(clause["all"] if "all" in clause else [clause["pred"]])
    return out


# --------------------------------------------------------------------------- #
# Candidate generation — greedy, residual-directed decision list               #
# --------------------------------------------------------------------------- #
def _hypothesis(clauses: List[Dict[str, Any]], default: Any) -> Dict[str, Any]:
    params = {"clauses": [dict(c) for c in clauses], "default": default}
    return {
        "kind": "rule",
        "rule_id": "DL_decision_list",
        "params": params,
        "name": f"decision list ({len(clauses)} clause{'s' if len(clauses) != 1 else ''})",
        "description": render_rule(params),
    }


def induce(train: List[Case], *, library: Optional[Librarian] = None,
           max_clauses: int = DEFAULT_MAX_CLAUSES) -> List[Dict[str, Any]]:
    """Propose candidate rules from TRAIN, simplest first (nested by length).

    Two stages, deliberately decoupled (the lesson of Level 4):
      1. *Discover clause conditions* — seed conjunctions from the highest-gain
         predicates and grow each toward purity (so a pure hierarchy clause like
         P1 & P2 is found even when an impure single predicate has higher gain).
      2. *Assemble* — lay the conditions into a decision list most-specific-first
         (purity order), so a pure, specific clause precedes a broad one that
         would otherwise shadow it. Decision lists are order-sensitive; this is
         what gives negation-by-ordering and recovers hierarchies.
    """
    n = len(train)
    true = [c["outcome"] for c in train]
    pool = mine_predicates(train, library)
    fire: List[List[bool]] = []
    specs: List[PredSpec] = []
    for spec in pool:
        pred = _build_predicate(spec)
        row = [bool(pred(c["events"])) for c in train]
        if any(row) and not all(row):     # a predicate that never/always fires is useless
            fire.append(row)
            specs.append(spec)

    support_floor = max(5, n // 40)
    # Two strategies; the arbiter chooses on the holdout. A nails the flat levels
    # (0-3) with minimal program length; B recovers hierarchies (Level 4) via
    # conjunctions + purity ordering. Diversity of proposals, MDL disposes.
    candidates = [_hypothesis([], _majority(true))]               # M1: trivial constant
    candidates += _greedy_list(fire, specs, true, n, support_floor, max_clauses)
    conditions = _candidate_conditions(fire, specs, true, n, support_floor)
    candidates += _assemble(conditions, fire, specs, true, n, support_floor, max_clauses)

    seen, deduped = set(), []
    for cand in candidates:
        key = cand["description"]
        if key not in seen:
            seen.add(key)
            deduped.append(cand)
    return deduped


def _greedy_list(fire: List[List[bool]], specs: List[PredSpec], true: List[Any],
                 n: int, support_floor: int, max_clauses: int) -> List[Dict[str, Any]]:
    """Strategy A: single-predicate, gain-greedy sequential covering (nested).

    Each clause is the single predicate that most increases the whole list's
    train accuracy (gain = correct in firing + majority of the rest - majority of
    the uncovered). Proven to recover the flat levels 0-3 with minimal length.
    """
    covered = [False] * n
    clauses: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for _ in range(max_clauses):
        uncovered = [i for i in range(n) if not covered[i]]      # M3: residual focus
        if not uncovered:
            break
        unc_counts = Counter(true[i] for i in uncovered)
        maj_uncovered = max(unc_counts.values())
        best = None
        for pi, row in enumerate(fire):
            firing = [i for i in uncovered if row[i]]
            if not firing or len(firing) == len(uncovered):
                continue
            fc = Counter(true[i] for i in firing)
            outcome = _argmax_label(fc)
            rest_max = max(unc_counts[l] - fc.get(l, 0) for l in unc_counts)
            gain = fc[outcome] + rest_max - maj_uncovered
            if gain <= 0:
                continue
            key = (gain, -_pred_complexity(specs[pi]))
            if best is None or key > best[0]:
                best = (key, pi, outcome, firing)
        if best is None:
            break
        _, pi, outcome, firing = best
        for i in firing:
            covered[i] = True
        clauses.append({"all": [specs[pi]], "outcome": outcome})
        remaining = [i for i in range(n) if not covered[i]]
        default = _majority([true[i] for i in remaining]) if remaining else outcome
        candidates.append(_hypothesis(clauses, default))
    return candidates


def _clause_stats(chosen: List[int], fire: List[List[bool]], true: List[Any],
                  domain: List[int]) -> Tuple[List[int], Any, int, float]:
    """Firing subset (within ``domain``), majority outcome, support, purity."""
    firing = [i for i in domain if all(fire[pi][i] for pi in chosen)]
    if not firing:
        return [], None, 0, 0.0
    counts = Counter(true[i] for i in firing)
    outcome = _argmax_label(counts)
    return firing, outcome, len(firing), counts[outcome] / len(firing)


def _candidate_conditions(fire: List[List[bool]], specs: List[PredSpec], true: List[Any],
                          n: int, support_floor: int) -> List[List[int]]:
    """Stage 1: seed from top-gain single predicates, grow each toward purity."""
    domain = list(range(n))
    unc_counts = Counter(true)
    maj = max(unc_counts.values())

    scored = []
    for pi, row in enumerate(fire):
        firing, outcome, support, _purity = _clause_stats([pi], fire, true, domain)
        if support < support_floor or support == n:
            continue
        fc = Counter(true[i] for i in firing)
        rest_max = max(unc_counts[l] - fc.get(l, 0) for l in unc_counts)
        gain = fc[outcome] + rest_max - maj
        if gain > 0:
            scored.append((gain, pi))
    scored.sort(key=lambda g: (-g[0], _pred_complexity(specs[g[1]])))

    conditions: List[List[int]] = []
    seen: set = set()
    for _gain, pi in scored[:12]:
        for cond in ([pi], _grow_purity([pi], fire, specs, true, n, support_floor)):
            key = tuple(sorted(cond))
            if key not in seen:
                seen.add(key)
                conditions.append(cond)
    return conditions


def _grow_purity(start: List[int], fire: List[List[bool]], specs: List[PredSpec],
                 true: List[Any], n: int, support_floor: int) -> List[int]:
    """AND predicates onto ``start`` while strictly increasing purity (precision)."""
    domain = list(range(n))
    chosen = list(start)
    _firing, _o, _supp, base_purity = _clause_stats(chosen, fire, true, domain)
    while len(chosen) < MAX_CONJ and base_purity < 1.0:
        best = None
        for pi, row in enumerate(fire):
            if pi in chosen:
                continue
            firing, _o, support, purity = _clause_stats(chosen + [pi], fire, true, domain)
            if support < support_floor or purity <= base_purity:
                continue
            # Among purity-improving refinements prefer the MOST GENERAL (highest
            # support): a high-support marker conjunction beats an overfit,
            # low-support successor that is pure only by coincidence.
            key = (purity, support, -_pred_complexity(specs[pi]))
            if best is None or key > best[0]:
                best = (key, pi)
        if best is None:
            break
        base_purity = best[0][0]
        chosen.append(best[1])
    return chosen


def _assemble(conditions: List[List[int]], fire: List[List[bool]], specs: List[PredSpec],
              true: List[Any], n: int, support_floor: int,
              max_clauses: int) -> List[Dict[str, Any]]:
    """Stage 2: purity-ordered sequential covering, emitting nested candidates."""
    domain = list(range(n))
    ranked = []
    for chosen in conditions:
        firing, _o, support, purity = _clause_stats(chosen, fire, true, domain)
        if support >= support_floor:
            ranked.append((purity, support, chosen, set(firing)))
    ranked.sort(key=lambda c: (-c[0], -c[1]))     # most-specific (purest) first

    covered = [False] * n
    clauses: List[Dict[str, Any]] = []
    candidates = [_hypothesis([], _majority(true))]   # M1: the trivial constant
    for _purity, _support, chosen, firing in ranked:
        if len(clauses) >= max_clauses:
            break
        uncovered = [i for i in range(n) if not covered[i]]
        new = [i for i in uncovered if i in firing]
        if len(new) < support_floor:
            continue
        cur_default = _majority([true[i] for i in uncovered])
        outcome = _argmax_label(Counter(true[i] for i in new))
        # Only add a clause that carves a *different* outcome than the running
        # default — a clause restating the default class wastes program length.
        benefit = (sum(1 for i in new if true[i] == outcome)
                   - sum(1 for i in new if true[i] == cur_default))
        if outcome == cur_default or benefit <= 0:
            continue
        for i in new:
            covered[i] = True
        clauses.append({"all": [specs[pi] for pi in chosen], "outcome": outcome})
        remaining = [i for i in range(n) if not covered[i]]
        default = _majority([true[i] for i in remaining]) if remaining else outcome
        candidates.append(_hypothesis(clauses, default))
    return candidates


# --------------------------------------------------------------------------- #
# Discover — propose (inducer) then dispose (arbiter on holdout)               #
# --------------------------------------------------------------------------- #
def discover(train: List[Case], test: List[Case], *,
             library: Optional[Librarian] = None,
             run_threshold: float = arb.DEFAULT_RUN_THRESHOLD_BITS,
             promote: bool = False, level_origin: Optional[str] = None,
             max_clauses: int = DEFAULT_MAX_CLAUSES) -> Dict[str, Any]:
    """Run the full loop: returns the best holdout-validated rule (or none)."""
    candidates = induce(train, library=library, max_clauses=max_clauses)
    ranked = []
    for cand in candidates:
        verdict = arb.evaluate(cand, train, test, run_threshold=run_threshold)
        ranked.append({"hypothesis": cand, "verdict": verdict})
    ranked.sort(key=lambda r: r["verdict"]["bits_saved"], reverse=True)

    accepted = [r for r in ranked if r["verdict"]["decision"] == "accept"]
    best = accepted[0] if accepted else None

    promotion = None
    if best is not None and promote and library is not None:
        # Re-adjudicate the winner so the arbiter owns the promotion decision.
        v = arb.adjudicate(best["hypothesis"], train, test, librarian=library,
                           run_threshold=run_threshold, level_origin=level_origin,
                           investigator="inducer", run_id=level_origin)
        promotion = v.get("promotion")
        best["verdict"]["promoted"] = v.get("promoted", False)

    return {
        "best": best,
        "ranked": ranked,
        "n_candidates": len(candidates),
        "found_rule": best is not None,
        "promotion": promotion,
    }


# --------------------------------------------------------------------------- #
# Legible rendering — the auditable deliverable                                #
# --------------------------------------------------------------------------- #
def render_pred(spec: PredSpec) -> str:
    p, prm = spec["pred"], spec.get("params", {})
    if p == "marker":
        return f"has({prm['marker']})"
    if p == "typed_successor":
        pairs = ", ".join(f"{a}->{b}" for a, b in prm["pairs"])
        return f"[{pairs}] within {prm['lag']}"
    if p == "count_at_least":
        return f"count({prm['type']})>={prm['n']}"
    if p == "attr_eq":
        return f"{prm['key']}=={prm['value']!r}"
    return f"{p}({prm})"


def render_condition(clause: Dict[str, Any]) -> str:
    preds = clause["all"] if "all" in clause else [clause["pred"]]
    return " AND ".join(render_pred(p) for p in preds)


def render_rule(params: Dict[str, Any]) -> str:
    parts = []
    for i, clause in enumerate(params["clauses"]):
        kw = "IF" if i == 0 else "ELIF"
        parts.append(f"{kw} {render_condition(clause)} THEN {clause['outcome']}")
    parts.append(f"ELSE {params['default']}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Induce a rule from a level's train split.")
    p.add_argument("--data", default="data")
    p.add_argument("--level", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=arb.DEFAULT_RUN_THRESHOLD_BITS)
    p.add_argument("--promote", action="store_true", help="promote the winner if it clears the library bar")
    p.add_argument("--library", default="library")
    p.add_argument("--use-library", action="store_true",
                   help="seed hypotheses with promoted library primitives (Mechanism 2)")
    args = p.parse_args(argv)

    train, test, gt = arb.load_split(args.data, args.level, args.seed)
    lib = Librarian(args.library) if (args.promote or args.use_library) else None
    result = discover(train, test, library=lib if args.use_library else None,
                      run_threshold=args.threshold, promote=args.promote,
                      level_origin=args.level)
    # Promotion needs a librarian even if not used for mining.
    if args.promote and result["best"] is not None and lib is not None and result["promotion"] is None:
        v = arb.adjudicate(result["best"]["hypothesis"], train, test, librarian=lib,
                           run_threshold=args.threshold, level_origin=args.level,
                           investigator="inducer", run_id=args.level)
        result["promotion"] = v.get("promotion")
        result["best"]["verdict"]["promoted"] = v.get("promoted", False)

    print(f"candidates evaluated : {result['n_candidates']}")
    if not result["found_rule"]:
        print("RESULT: no rule compresses the holdout — reporting NOTHING (no hallucination).")
        return 1

    best = result["best"]
    v = best["verdict"]
    print(f"RESULT: {best['hypothesis']['name']}")
    print(f"  rule       : {best['hypothesis']['description']}")
    print(f"  bits_saved : {v['bits_saved']:.2f}   test_acc {v['test_accuracy']:.3f}")
    if v.get("promoted"):
        print(f"  promoted   : {result['promotion']}")

    # Offline ground-truth diagnostic (NOT used for any decision).
    try:
        from .metrics import recovered
        rec = recovered(args.level, gt, best["hypothesis"])
        verdict = "EQUIVALENT to planted rule" if rec["equivalent"] else "NOT equivalent"
        print(f"  recovery   : {verdict} (agreement {rec['agreement']:.3f} on a fresh sample)")
    except (ValueError, KeyError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
