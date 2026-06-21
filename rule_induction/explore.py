"""The investigator's TRAIN-ONLY scratchpad (Mechanism: smarter proposer).

A good investigator doesn't eyeball the data — it *computes*: "is max nesting depth
always 3 for accepts?", "which event types correlate with the outcome?", "what do
my residual cases have in common?". This instrument lets the agent run arbitrary
analysis code over the **train** split to form better hypotheses.

The honesty boundary is enforced by construction, not by trust: the analysis code
runs in the same hostile sandbox as a hypothesis (no ``open``/``__import__``), and
this tool hands it **only** the train cases — it never loads ``test.jsonl`` or
``ground_truth.json``. So the agent can compute anything it likes; it still cannot
peek at the answer, and the arbiter remains the blind judge on the holdout.

The scratchpad code defines one function::

    def analyze(train):
        # train: list of case dicts {case_id, events:[{type,attrs}], outcome, ...}
        # return any JSON-serialisable value (dict/list/number/string)
        ...
        return {...}

Usage::

    python -m rule_induction.explore --dataset data/grammar_hard --code scratch.py
    python -m rule_induction.explore --data data --level level5 --seed 0 --code -   # stdin
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from . import sandbox
from .present import load_train, load_train_dir

Case = Dict[str, Any]


def explore(train: List[Case], source: str, **sandbox_opts) -> Any:
    """Run the agent's ``analyze(train)`` over the train cases, in the sandbox."""
    return sandbox.run_analysis(source, train, **sandbox_opts)


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run TRAIN-ONLY analysis code (the investigator's scratchpad).")
    p.add_argument("--data", default="data")
    p.add_argument("--level", help="bench level (e.g. level5); omit if using --dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", help="a folder holding YOUR own train.jsonl (overrides --level)")
    p.add_argument("--code", required=True,
                   help="file defining analyze(train); use '-' to read from stdin")
    args = p.parse_args(argv)

    if args.dataset:
        train = load_train_dir(args.dataset)
        tag = args.dataset
    else:
        if not args.level:
            p.error("provide --level (bench) or --dataset (your own folder)")
        train = load_train(args.data, args.level, args.seed)
        tag = f"{args.level} seed {args.seed}"

    source = sys.stdin.read() if args.code == "-" else open(args.code, encoding="utf-8").read()

    print(f"=== Scratchpad: {tag} (TRAIN ONLY, n={len(train)}) ===")
    print("(sandboxed: no file access — test.jsonl / ground_truth.json are unreachable)\n")
    try:
        result = explore(train, source)
    except sandbox.SandboxError as exc:
        print(f"scratchpad error: {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
