"""Demo — the agent invents a primitive, then reuses it cheaply (two-part MDL).

The whole point of the abstraction question: when no fixed-grammar predicate
compresses the data, the agent *invents* one; once it earns its place in the
library it becomes shared vocabulary, so the next dataset reuses it for a fraction
of the bits. Inventing is expensive; reuse is cheap — abstraction earns its keep.

Two datasets with the SAME abstract structure in DIFFERENT vocabularies:
  * A: outcome by the sign of (#up − #down)   -> flag / block / review(tie)
  * B: outcome by the sign of (#win − #loss)  -> same structure, new words

The fixed grammar's ``count_at_least`` compares a count to a *constant*; it cannot
compare *two* counts, so it cannot express this rule. The invented predicate
``count_gt(events, a, b)`` can. Run::

    python -m rule_induction.demo_invent
"""

from __future__ import annotations

import random
import tempfile

from . import arbiter
from .dataset import split_train_test
from .librarian import Librarian
from .model import event, retime

# The predicate the agent invents — a comparison between two counts, outside the
# fixed grammar. (In the real system this source is what the Claude agent writes.)
COUNT_GT = '''
def count_gt(events, a, b):
    return sum(e["type"] == a for e in events) > sum(e["type"] == b for e in events)
'''


def _gen(seed: int, n: int, a_tok: str, b_tok: str):
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        nu, nd = rng.randint(0, 5), rng.randint(0, 5)
        body = [event(0, a_tok) for _ in range(nu)] + [event(0, b_tok) for _ in range(nd)]
        for _ in range(rng.randint(0, 3)):                 # neutral noise
            body.append(event(0, rng.choice(["evt_P", "evt_Q", "evt_R"])))
        rng.shuffle(body)
        events = retime([event(0, "open", {})] + body + [event(0, "close")])
        outcome = "flag" if nu > nd else ("block" if nd > nu else "review")
        cases.append({"case_id": f"c_{i:05d}", "events": events,
                      "outcome": outcome, "level": "invent"})
    return cases


def _composed(a_tok: str, b_tok: str, desc: str):
    return {
        "kind": "composed",
        "name": "count_gt",
        "description": desc,
        "primitives": {"count_gt": COUNT_GT},
        "clauses": [
            {"all": [{"prim": "count_gt", "params": {"a": a_tok, "b": b_tok}}], "outcome": "flag"},
            {"all": [{"prim": "count_gt", "params": {"a": b_tok, "b": a_tok}}], "outcome": "block"},
        ],
        "default": "review",
    }


def main() -> int:
    libroot = tempfile.mkdtemp(prefix="eureka_invent_")
    lib = Librarian(libroot)
    print(f"fresh library: {libroot}\n")

    # ----- Dataset A: invent the primitive -------------------------------- #
    A = _gen(0, 1500, "up", "down")
    trainA, testA = split_train_test(A, 0.6, 0)

    baseline = {"kind": "rule", "rule_id": "DL_decision_list", "params": {
        "clauses": [
            {"all": [{"pred": "count_at_least", "params": {"type": "up", "n": 3}}], "outcome": "flag"},
            {"all": [{"pred": "count_at_least", "params": {"type": "down", "n": 3}}], "outcome": "block"}],
        "default": "review"}}
    vb = arbiter.evaluate(baseline, trainA, testA)
    print("A · fixed grammar (count_at_least vs constant) : "
          f"acc={vb['test_accuracy']:.3f}  bits_saved={vb['bits_saved']:+7.1f}  -> {vb['decision'].upper()}")

    known0 = lib.known_primitive_names()
    vA = arbiter.adjudicate(_composed("up", "down", "flag/block by which token is more frequent"),
                            trainA, testA, librarian=lib, known_primitives=known0,
                            level_origin="invent/A", investigator="analogy")
    print("A · INVENT count_gt (new primitive)            : "
          f"acc={vA['test_accuracy']:.3f}  bits_saved={vA['bits_saved']:+7.1f}  -> {vA['decision'].upper()}"
          f"   [L_program={vA['program_bits']:.0f} bits, promoted={vA['promoted']}]")

    # ----- Dataset B: reuse it on a new vocabulary ------------------------ #
    B = _gen(1, 1500, "win", "loss")
    trainB, testB = split_train_test(B, 0.6, 1)
    known1 = lib.known_primitive_names()              # now contains count_gt
    vB = arbiter.adjudicate(_composed("win", "loss", "same structure, new vocabulary"),
                            trainB, testB, librarian=lib, known_primitives=known1,
                            level_origin="invent/B", investigator="analogy")
    print("B · REUSE count_gt (now shared vocabulary)     : "
          f"acc={vB['test_accuracy']:.3f}  bits_saved={vB['bits_saved']:+7.1f}  -> {vB['decision'].upper()}"
          f"   [L_program={vB['program_bits']:.0f} bits]")

    saved = vA["program_bits"] - vB["program_bits"]
    print(f"\nAmortization: inventing cost {vA['program_bits']:.0f} program-bits; "
          f"reusing cost only {vB['program_bits']:.0f}. "
          f"Having the abstraction saved {saved:.0f} bits on B.")
    print("The fixed grammar could not express the rule; the invented primitive could, "
          "and it transferred across vocabularies — analogy + invention, both verified on holdout.")

    print("\nLibrary now holds:")
    for r in lib.list(kind="primitive"):
        print(f"  [primitive] {r['name']:<10} {r['bits_saved']:+.1f} bits — {r['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
