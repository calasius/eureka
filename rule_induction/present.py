"""Investigator's view of a level — the honest reading surface (Section 5).

This is what an *investigator* (a Claude agent running the `investigator` skill)
is allowed to see when proposing hypotheses: **train only**. It deliberately
never loads `test.jsonl` outcomes or `ground_truth.json`, so a human-or-LLM
investigator cannot peek at the answer. The arbiter — deterministic code scoring
MDL on the holdout — remains the only thing that judges a proposal.

Two views:
  * `present` — the train traces in a compact, readable form, plus a summary of
    the event vocabulary, attributes, and label distribution. This is the raw
    material for proposing a rule.
  * `residuals` — given a candidate hypothesis, the TRAIN cases it currently gets
    wrong. This drives surprise-directed search (Mechanism 3): propose where the
    current best rule fails, not where it already predicts well.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

Case = Dict[str, Any]


def _read_jsonl(path: str) -> List[Case]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_train(data_dir: str, level: str, seed: int) -> List[Case]:
    """Train cases only — the investigator never sees test outcomes or ground truth."""
    return _read_jsonl(os.path.join(data_dir, level, f"seed_{seed}", "train.jsonl"))


def load_train_dir(dataset_dir: str) -> List[Case]:
    """Train cases from a flat dataset directory holding train.jsonl directly."""
    return _read_jsonl(os.path.join(dataset_dir, "train.jsonl"))


def render_events(events: List[Dict[str, Any]]) -> str:
    parts = []
    for e in events:
        attrs = e.get("attrs") or {}
        if attrs:
            kv = ",".join(f"{k}={v}" for k, v in attrs.items())
            parts.append(f"{e['type']}({kv})")
        else:
            parts.append(e["type"])
    return " ".join(parts)


def summarize(train: List[Case]) -> Dict[str, Any]:
    types: Counter = Counter()
    attr_values: Dict[str, set] = {}
    outcomes: Counter = Counter()
    for c in train:
        outcomes[c["outcome"]] += 1
        for e in c["events"]:
            types[e["type"]] += 1
            for k, v in (e.get("attrs") or {}).items():
                attr_values.setdefault(k, set()).add(v)
    return {
        "n_train": len(train),
        "event_types": dict(types.most_common()),
        "attributes": {k: sorted(map(str, v)) for k, v in sorted(attr_values.items())},
        "outcomes": dict(sorted(outcomes.items())),
    }


def residuals(train: List[Case], hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """TRAIN cases a hypothesis mispredicts — the surprise-directed agenda."""
    from . import arbiter as arb   # local import to avoid a cycle
    preds = arb._predict(hypothesis, train)
    out = []
    for c, p in zip(train, preds):
        if p != c["outcome"]:
            out.append({"case_id": c["case_id"], "predicted": p,
                        "actual": c["outcome"], "events": render_events(c["events"])})
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Investigator's train-only view of a level.")
    p.add_argument("--data", default="data")
    p.add_argument("--level", help="bench level (e.g. level4); omit if using --dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", help="a folder holding YOUR own train.jsonl (overrides --level)")
    p.add_argument("--limit", type=int, default=40, help="max cases to print")
    p.add_argument("--residuals", help="JSON hypothesis file; print the train cases it gets wrong")
    args = p.parse_args(argv)

    if args.dataset:
        train = load_train_dir(args.dataset)
        tag = args.dataset
    else:
        if not args.level:
            p.error("provide --level (bench) or --dataset (your own folder)")
        train = load_train(args.data, args.level, args.seed)
        tag = f"{args.level} seed {args.seed}"
    summary = summarize(train)

    print(f"=== Investigator view: {tag} (TRAIN ONLY) ===")
    print(f"n_train={summary['n_train']}  outcomes={summary['outcomes']}")
    print(f"event_types={summary['event_types']}")
    print(f"attributes={summary['attributes']}")

    if args.residuals:
        with open(args.residuals, encoding="utf-8") as fh:
            hyp = json.load(fh)
        res = residuals(train, hyp)
        print(f"\n--- residuals: {len(res)}/{len(train)} train cases mispredicted ---")
        for r in res[:args.limit]:
            print(f"  {r['case_id']}: pred={r['predicted']} actual={r['actual']} | {r['events'][:90]}")
        return 0

    print("\n--- train traces (events => outcome) ---")
    for c in train[:args.limit]:
        print(f"  {c['case_id']}: {render_events(c['events'])[:100]}  =>  {c['outcome']}")
    if len(train) > args.limit:
        print(f"  ... ({len(train) - args.limit} more)")
    print("\nPropose a hypothesis (rule spec or code), then verify with:")
    if args.dataset:
        print(f"  python -m rule_induction.arbiter --dataset {args.dataset} --hypothesis hyp.json")
    else:
        print(f"  python -m rule_induction.arbiter --data {args.data} --level {args.level} "
              f"--seed {args.seed} --hypothesis hyp.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
