"""The Arbiter — Skill 2, the honesty anchor.

Given a candidate hypothesis and a level's data, the arbiter:

  1. Loads the **holdout** split (train + test). The inducer only ever saw train.
  2. Produces predictions:
       * ``rule`` hypotheses are symbolic (built from our own registry) and run
         in-process — trusted.
       * ``code`` hypotheses are **hostile-until-clean**: run in the sandbox; any
         dirty run is an automatic rejection.
  3. Scores **MDL compression on test** (program length included).
  4. Accepts iff ``bits_saved >= run_threshold``; on acceptance above the
     (stricter) **library threshold**, promotes the rule to the Librarian.

Critically, the arbiter never reads ``ground_truth.json`` to make predictions —
it judges hypotheses against held-out *data*, not against the planted answer.
The planted rule is only used by the offline metrics in ``metrics.py``.

Hypothesis JSON::

    {"kind": "rule", "rule_id": "R1_typed_successor", "params": {...},
     "name": "...", "description": "..."}

    {"kind": "code", "source": "def predict(events):\n    ...",
     "name": "...", "description": "..."}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import mdl, primitives, sandbox
from .librarian import Librarian, PromotionRejected
from .rules import make_labeler

DEFAULT_RUN_THRESHOLD_BITS = 2.0   # low bar to "use once"; library threshold is stricter

Case = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def _read_jsonl(path: str) -> List[Case]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_split(data_dir: str, level: str, seed: int
               ) -> Tuple[List[Case], List[Case], Dict[str, Any]]:
    """Return (train, test, ground_truth) for one (level, seed)."""
    seed_dir = os.path.join(data_dir, level, f"seed_{seed}")
    train = _read_jsonl(os.path.join(seed_dir, "train.jsonl"))
    test = _read_jsonl(os.path.join(seed_dir, "test.jsonl"))
    with open(os.path.join(seed_dir, "ground_truth.json"), encoding="utf-8") as fh:
        gt = json.load(fh)
    return train, test, gt


def load_dataset(dataset_dir: str) -> Tuple[List[Case], List[Case], Optional[Dict[str, Any]]]:
    """Load YOUR own dataset: a folder with train.jsonl + test.jsonl (ground_truth optional)."""
    train = _read_jsonl(os.path.join(dataset_dir, "train.jsonl"))
    test = _read_jsonl(os.path.join(dataset_dir, "test.jsonl"))
    gt_path = os.path.join(dataset_dir, "ground_truth.json")
    gt = None
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as fh:
            gt = json.load(fh)
    return train, test, gt


# --------------------------------------------------------------------------- #
# Prediction                                                                   #
# --------------------------------------------------------------------------- #
def _predict(hypothesis: Dict[str, Any], cases: List[Case], **sandbox_opts) -> List[Any]:
    kind = hypothesis.get("kind")
    if kind == "rule":
        labeler = make_labeler(hypothesis["rule_id"], hypothesis.get("params", {}))
        return [labeler(c["events"]) for c in cases]
    if kind == "code":
        return sandbox.run_code(hypothesis["source"], cases, **sandbox_opts)
    if kind == "composed":
        # Invented primitives are LLM-generated -> sandbox, never in-process.
        source = primitives.compile_composed(hypothesis)
        return sandbox.run_code(source, cases, **sandbox_opts)
    raise ValueError(f"unknown hypothesis kind: {kind!r}")


def _program_bits(hypothesis: Dict[str, Any],
                  known_primitives: frozenset = frozenset()) -> float:
    kind = hypothesis.get("kind")
    if kind == "rule":
        return mdl.program_bits_spec(hypothesis.get("params", {}))
    if kind == "composed":
        return primitives.program_bits_composed(hypothesis, known_primitives)
    return mdl.program_bits_code(hypothesis["source"])


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate(hypothesis: Dict[str, Any], train: List[Case], test: List[Case], *,
             run_threshold: float = DEFAULT_RUN_THRESHOLD_BITS,
             known_primitives: frozenset = frozenset(),
             **sandbox_opts) -> Dict[str, Any]:
    """Score a hypothesis on the holdout. Returns the verdict (no promotion).

    ``known_primitives`` are invented predicates already in the library; a composed
    hypothesis reusing them pays only a pointer, not their full description length.
    """
    try:
        train_pred = _predict(hypothesis, train, **sandbox_opts)
        test_pred = _predict(hypothesis, test, **sandbox_opts)
    except sandbox.SandboxError as exc:
        return {
            "decision": "reject",
            "reason": f"sandbox: {exc}",
            "bits_saved": float("-inf"),
            "sandbox_clean": False,
        }

    train_true = [c["outcome"] for c in train]
    test_true = [c["outcome"] for c in test]
    result = mdl.score(train_true, train_pred, test_true, test_pred,
                       _program_bits(hypothesis, known_primitives))
    accept = result["bits_saved"] >= run_threshold
    result.update({
        "decision": "accept" if accept else "reject",
        "reason": ("compresses holdout" if accept
                   else f"bits_saved {result['bits_saved']:.2f} < run threshold {run_threshold:.2f}"),
        "run_threshold": run_threshold,
        "sandbox_clean": True,
    })
    return result


def _to_library_entry(hypothesis: Dict[str, Any], verdict: Dict[str, Any],
                      *, level_origin: Optional[str], run_id: Optional[str],
                      investigator: Optional[str]) -> Dict[str, Any]:
    is_rule = hypothesis.get("kind") == "rule"
    entry = {
        "name": hypothesis.get("name") or hypothesis.get("rule_id") or "hypothesis",
        "kind": "rule" if is_rule else "function",
        "description": hypothesis.get("description", ""),
        "rule_id": hypothesis.get("rule_id"),
        "spec": hypothesis.get("params", {}) if is_rule else {},
        "program": None if is_rule else hypothesis.get("source"),
        "mdl": {
            "bits_saved": verdict["bits_saved"],
            "compresses": f"{level_origin or 'data'} outcomes (holdout)",
            "score": {k: verdict[k] for k in
                      ("l_null", "program_bits", "l_data_given_h", "test_accuracy")},
        },
        "provenance": {"run_id": run_id, "investigator": investigator,
                       "level_origin": level_origin, "evaluated_on": "holdout"},
    }
    return entry


def _promote_primitives(hypothesis: Dict[str, Any], verdict: Dict[str, Any],
                        librarian: Librarian, known_primitives: frozenset, *,
                        level_origin: Optional[str], run_id: Optional[str],
                        investigator: Optional[str]) -> List[Dict[str, Any]]:
    """Admit the freshly-invented predicates to the library as reusable vocabulary."""
    promoted = []
    for name in primitives.new_primitive_names(hypothesis, known_primitives):
        entry = {
            "name": name,
            "kind": "primitive",
            "description": (f"Invented predicate {name}(events, ...) — "
                            f"{hypothesis.get('description', '')}").strip(),
            "program": hypothesis["primitives"][name],
            "mdl": {"bits_saved": verdict["bits_saved"],
                    "compresses": f"{level_origin or 'data'} outcomes (holdout)",
                    "score": {k: verdict.get(k) for k in
                              ("l_null", "program_bits", "l_data_given_h", "test_accuracy")}},
            "provenance": {"run_id": run_id, "investigator": investigator,
                           "level_origin": level_origin, "evaluated_on": "holdout"},
        }
        try:
            record = librarian.promote(entry)
            promoted.append({"id": record["id"], "version": record["version"],
                             "commit": record.get("commit")})
        except PromotionRejected as exc:
            promoted.append({"name": name, "rejected": str(exc)})
    return promoted


def adjudicate(hypothesis: Dict[str, Any], train: List[Case], test: List[Case], *,
               librarian: Optional[Librarian] = None,
               run_threshold: float = DEFAULT_RUN_THRESHOLD_BITS,
               known_primitives: frozenset = frozenset(),
               level_origin: Optional[str] = None,
               run_id: Optional[str] = None,
               investigator: Optional[str] = None,
               **sandbox_opts) -> Dict[str, Any]:
    """Evaluate and, if accepted and a librarian is given, attempt promotion."""
    verdict = evaluate(hypothesis, train, test, run_threshold=run_threshold,
                       known_primitives=known_primitives, **sandbox_opts)
    verdict["promoted"] = False
    if verdict["decision"] == "accept" and librarian is not None:
        if hypothesis.get("kind") == "composed":
            # The abstraction worth keeping is the invented primitive, not the rule.
            promos = _promote_primitives(hypothesis, verdict, librarian, known_primitives,
                                         level_origin=level_origin, run_id=run_id,
                                         investigator=investigator)
            verdict["promoted"] = any("id" in p for p in promos)
            verdict["promotion"] = {"primitives": promos}
        else:
            entry = _to_library_entry(hypothesis, verdict, level_origin=level_origin,
                                      run_id=run_id, investigator=investigator)
            try:
                record = librarian.promote(entry)
                verdict["promoted"] = True
                verdict["promotion"] = {"id": record["id"], "version": record["version"],
                                        "commit": record.get("commit")}
            except PromotionRejected as exc:
                verdict["promotion"] = {"rejected": str(exc)}
    return verdict


def run(data_dir: str, level: str, seed: int, hypothesis: Dict[str, Any], *,
        library_root: Optional[str] = None, promote: bool = False,
        run_threshold: float = DEFAULT_RUN_THRESHOLD_BITS,
        **sandbox_opts) -> Dict[str, Any]:
    """Load a level's holdout and adjudicate a hypothesis end-to-end."""
    train, test, _gt = load_split(data_dir, level, seed)
    lib = Librarian(library_root) if (promote and library_root) else None
    known = lib.known_primitive_names() if lib else frozenset()
    return adjudicate(hypothesis, train, test, librarian=lib,
                      run_threshold=run_threshold, known_primitives=known,
                      level_origin=level, run_id=f"{level}_seed{seed}", **sandbox_opts)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Adjudicate a hypothesis against a level's holdout.")
    p.add_argument("--data", default="data", help="bench dataset dir [default: data]")
    p.add_argument("--level", help="bench level; omit if using --dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", help="a folder with YOUR own train.jsonl + test.jsonl (overrides --level)")
    p.add_argument("--hypothesis", required=True, help="JSON file describing the hypothesis")
    p.add_argument("--threshold", type=float, default=DEFAULT_RUN_THRESHOLD_BITS,
                   help="run-acceptance threshold in bits")
    p.add_argument("--promote", action="store_true", help="promote to the library if accepted")
    p.add_argument("--library", default="library", help="library root for promotion")
    args = p.parse_args(argv)

    with open(args.hypothesis, encoding="utf-8") as fh:
        hypothesis = json.load(fh)

    if args.dataset:
        train, test, _gt = load_dataset(args.dataset)
        lib = Librarian(args.library) if args.promote else None
        known = lib.known_primitive_names() if lib else frozenset()
        verdict = adjudicate(hypothesis, train, test, librarian=lib,
                             run_threshold=args.threshold, known_primitives=known,
                             level_origin=args.dataset, run_id=args.dataset)
    else:
        if not args.level:
            p.error("provide --level (bench) or --dataset (your own folder)")
        verdict = run(args.data, args.level, args.seed, hypothesis,
                      library_root=args.library, promote=args.promote,
                      run_threshold=args.threshold)

    print(f"decision     : {verdict['decision'].upper()}  ({verdict['reason']})")
    if "bits_saved" in verdict and verdict["bits_saved"] != float("-inf"):
        print(f"bits_saved   : {verdict['bits_saved']:.2f}")
        print(f"  L_null     : {verdict['l_null']:.2f}")
        print(f"  L_program  : {verdict['program_bits']:.2f}")
        print(f"  L_data|H   : {verdict['l_data_given_h']:.2f}")
        print(f"test_acc     : {verdict['test_accuracy']:.3f}   "
              f"(train_acc {verdict['train_accuracy']:.3f})")
    if "promotion" in verdict and "primitives" in verdict["promotion"]:
        for p in verdict["promotion"]["primitives"]:
            if "id" in p:
                print(f"primitive    : promoted {p['id']} (v{p['version']}, commit {p.get('commit')})")
            else:
                print(f"primitive    : rejected {p.get('name')} — {p.get('rejected')}")
    elif verdict.get("promoted"):
        promo = verdict["promotion"]
        print(f"promoted     : {promo['id']} (v{promo['version']}, commit {promo.get('commit')})")
    elif "promotion" in verdict and "rejected" in verdict["promotion"]:
        print(f"promotion    : rejected — {verdict['promotion']['rejected']}")
    return 0 if verdict["decision"] == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
