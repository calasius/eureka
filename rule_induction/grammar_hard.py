"""A HARD grammar-induction dataset: typed nesting + exact depth + count agreement.

Each case is a string over a meaningless alphabet; the outcome is ``accept`` iff
ALL THREE coupled constraints hold:

  1. **Typed well-nesting** — every opener closed by its OWN type, properly nested
     (``( [ ] )`` ok, ``( [ ) ]`` invalid). Two bracket types:
     round ``kx``/``kz`` and square ``wp``/``wq``. Needs a STACK — context-free,
     beyond anything ``count_at_least`` can express.
  2. **Exact maximum depth == 3** — a global constraint coupling the whole string.
  3. **Equal pair counts** — #round-pairs == #square-pairs.

``n1``/``n2`` are noise tokens, ignored by the rule. The negatives are *hard*: many
satisfy two of the three constraints, so any partial hypothesis (balance-only,
count-only, nesting-only, depth-only) is wrong on a chunk of the data.

This is meant to sit at the FRONTIER of what the system recovers — the honest point
of a hard demo. A correct hypothesis must invent a small pushdown recogniser; it is
an expensive abstraction (≈700 program-bits), so it only clears MDL with enough
holdout evidence (and would amortise across datasets via the library).

Layout written (flat ``--dataset`` form)::

    <out>/train.jsonl  <out>/test.jsonl  <out>/ground_truth.json  (sealed answer)

CLI::

    python -m rule_induction.grammar_hard --out data/grammar_hard --n 2400 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

from .dataset import split_train_test
from .model import event, retime

Case = Dict[str, Any]

PAIRS = {"kx": "kz", "wp": "wq"}          # opener -> closer, per type
CLOSERS = {c: o for o, c in PAIRS.items()}
NOISE = ["n1", "n2"]
TARGET_DEPTH = 3


# --------------------------------------------------------------------------- #
# The hidden rule (single source of truth for the label)                       #
# --------------------------------------------------------------------------- #
def rule_accept(events: List[Dict[str, Any]]) -> bool:
    stack: List[str] = []
    maxd = 0
    counts = {"kx": 0, "wp": 0}
    for e in events:
        t = e["type"]
        if t in PAIRS:
            stack.append(t)
            maxd = max(maxd, len(stack))
        elif t in CLOSERS:
            if not stack or stack[-1] != CLOSERS[t]:
                return False
            counts[stack.pop()] += 1
    if stack:
        return False
    if maxd != TARGET_DEPTH:
        return False
    return counts["kx"] == counts["wp"]


def _maxdepth(tokens: List[str]) -> int:
    stack, maxd = [], 0
    for t in tokens:
        if t in PAIRS:
            stack.append(t); maxd = max(maxd, len(stack))
        elif t in CLOSERS and stack:
            stack.pop()
    return maxd


# --------------------------------------------------------------------------- #
# Sequence synthesis                                                            #
# --------------------------------------------------------------------------- #
def _random_valid(rng: random.Random, kr: int, ks: int, depth_cap: int) -> List[str]:
    """A typed-well-nested token list: kr round-pairs, ks square-pairs, depth<=cap."""
    open_left = {"kx": kr, "wp": ks}
    stack: List[str] = []
    out: List[str] = []
    while open_left["kx"] + open_left["wp"] > 0 or stack:
        can_open = ([t for t in open_left if open_left[t] > 0]
                    if len(stack) < depth_cap else [])
        acts = (["open"] if can_open else []) + (["close"] if stack else [])
        if not acts:
            break
        if rng.choice(acts) == "open":
            t = rng.choice(can_open)
            open_left[t] -= 1
            stack.append(t)
            out.append(t)
        else:
            out.append(PAIRS[stack.pop()])
    return out


def _valid_at_depth(rng: random.Random, kr: int, ks: int, depth: int) -> List[str]:
    """A valid sequence whose max depth is exactly ``depth`` (best effort)."""
    best: List[str] = []
    for _ in range(80):
        toks = _random_valid(rng, kr, ks, depth)
        if _maxdepth(toks) == depth:
            return toks
        best = toks
    return best


def _gen_tokens(rng: random.Random) -> List[str]:
    """Produce one candidate token list from a mix of positive/negative flavours."""
    flavour = rng.choice([
        "pos", "pos", "pos",                 # valid: depth 3, equal counts
        "wrong_depth", "wrong_count",         # valid nesting, one constraint broken
        "type_mismatch", "unbalanced", "close_first",
    ])
    k = rng.randint(2, 4)
    if flavour == "pos":
        return _valid_at_depth(rng, k, k, TARGET_DEPTH)
    if flavour == "wrong_depth":
        return _valid_at_depth(rng, k, k, rng.choice([2, 4]))      # depth != 3
    if flavour == "wrong_count":
        ks = max(1, k + rng.choice([-1, 1, 2]))
        if ks == k:
            ks = k + 1
        return _valid_at_depth(rng, k, ks, TARGET_DEPTH)           # equal-count broken
    if flavour == "type_mismatch":
        toks = _valid_at_depth(rng, k, k, TARGET_DEPTH)
        idx = [i for i, t in enumerate(toks) if t in CLOSERS]
        if idx:
            i = rng.choice(idx)
            toks[i] = "wq" if toks[i] == "kz" else "kz"            # flip a closer's type
        return toks
    if flavour == "unbalanced":
        toks = _valid_at_depth(rng, k, k, TARGET_DEPTH)
        idx = [i for i, t in enumerate(toks) if t in CLOSERS]
        if idx:
            toks.pop(rng.choice(idx))                              # drop a closer
        return toks
    # close_first
    toks = _valid_at_depth(rng, k, k, TARGET_DEPTH)
    return [rng.choice(["kz", "wq"])] + toks                       # closer before any opener


def _with_noise(rng: random.Random, tokens: List[str]):
    toks = list(tokens)
    for _ in range(rng.randint(0, 3)):
        toks.insert(rng.randint(0, len(toks)), rng.choice(NOISE))
    return retime([event(0, t) for t in toks])


def generate(n: int, seed: int) -> List[Case]:
    """Build a class-balanced dataset, labelling every case by the true rule."""
    rng = random.Random(seed)
    pos, neg = [], []
    guard = 0
    while (len(pos) < n // 2 or len(neg) < n // 2) and guard < n * 50:
        guard += 1
        events = _with_noise(rng, _gen_tokens(rng))
        accept = rule_accept(events)
        bucket = pos if accept else neg
        if len(bucket) >= n // 2:
            continue
        i = len(pos) + len(neg)
        bucket.append({"case_id": f"g_{i:05d}", "events": events,
                       "outcome": "accept" if accept else "reject", "level": "grammar_hard"})
    cases = pos + neg
    rng.shuffle(cases)
    for i, c in enumerate(cases):
        c["case_id"] = f"g_{i:05d}"
    return cases


# The compact recogniser a correct hypothesis converges to — the SEALED answer,
# expressed as a composed hypothesis (invented primitive + decision list).
GRAMMAR_PRIMITIVE = '''
def grammar_ok(events):
    pairs = {"kx": "kz", "wp": "wq"}
    closers = {"kz": "kx", "wq": "wp"}
    stack = []
    maxd = cr = cs = 0
    for e in events:
        t = e["type"]
        if t in pairs:
            stack.append(t)
            if len(stack) > maxd:
                maxd = len(stack)
        elif t in closers:
            if not stack or stack[-1] != closers[t]:
                return False
            top = stack.pop()
            if top == "kx":
                cr += 1
            else:
                cs += 1
    if stack:
        return False
    if maxd != 3:
        return False
    return cr == cs
'''


def reference_hypothesis() -> Dict[str, Any]:
    return {
        "kind": "composed",
        "name": "grammar_ok",
        "description": "typed-well-nested AND max-depth==3 AND #round-pairs==#square-pairs",
        "primitives": {"grammar_ok": GRAMMAR_PRIMITIVE},
        "clauses": [{"all": [{"prim": "grammar_ok", "params": {}}], "outcome": "accept"}],
        "default": "reject",
    }


# --------------------------------------------------------------------------- #
# Writing                                                                       #
# --------------------------------------------------------------------------- #
def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset(out_dir: str, *, n: int, seed: int, ratio: float,
                  with_ground_truth: bool = True) -> Dict[str, Any]:
    cases = generate(n, seed)
    train, test = split_train_test(cases, ratio, seed)
    os.makedirs(out_dir, exist_ok=True)
    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    _write_jsonl(os.path.join(out_dir, "test.jsonl"), test)
    dist = dict(sorted(Counter(c["outcome"] for c in cases).items()))
    if with_ground_truth:
        gt = {
            "planted_rule_id": "GRAMMAR_typed_nesting_depth_agreement",
            "params": {"pairs": PAIRS, "noise": NOISE, "exact_max_depth": TARGET_DEPTH,
                       "constraint": "count(round-pairs) == count(square-pairs)"},
            "description": ("accept iff (1) typed-well-nested, (2) max nesting depth "
                            "EXACTLY 3, (3) #round-pairs == #square-pairs; n1,n2 ignored"),
            "label_distribution": dist, "n_train": len(train), "n_test": len(test),
            "reference_solution": reference_hypothesis(),
            "scorer_note": "Outcome == rule_accept(events). Hard negatives satisfy 2 of 3 constraints.",
        }
        with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as fh:
            json.dump(gt, fh, indent=2, ensure_ascii=False)
    return {"out_dir": out_dir, "n_train": len(train), "n_test": len(test),
            "label_distribution": dist}


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the HARD typed-nesting grammar dataset.")
    p.add_argument("--out", default="data/grammar_hard")
    p.add_argument("--n", type=int, default=2400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument("--no-ground-truth", action="store_true")
    args = p.parse_args(argv)
    info = write_dataset(args.out, n=args.n, seed=args.seed, ratio=args.ratio,
                         with_ground_truth=not args.no_ground_truth)
    print(f"wrote {info['n_train']} train + {info['n_test']} test to {info['out_dir']}/")
    print(f"label distribution: {info['label_distribution']}")
    print(f"\nnext:\n  python -m rule_induction.present --dataset {info['out_dir']}")
    print(f"  # then in a Claude session:  /investigator --dataset {info['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
