---
name: inducer
description: >-
  Discover the hidden generative rule behind an event trace and return it as an
  explicit, auditable rule (not a prediction). Mines a predicate pool from the
  data, generates candidate rules (simplicity-ordered, residual-directed,
  recombining library primitives), and hands them to the arbiter, which validates
  on a holdout. Use when there is a sequence/trace and you want the rule that
  generated it.
---

# Inducer — the orchestrator (Skill 1)

This is the **propose** half of "LLM proposes / code verifies". It turns event
traces into an explicit rule by generating the few hypotheses worth testing out
of an infinite space, then letting the arbiter judge them on held-out data.

## The honesty boundary (read first)

The inducer reads **train only** to *propose* hypotheses — it is allowed to fit
train. The **arbiter** then scores each candidate on the **holdout** (with the
program length included) and *selects* the winner. The inducer never inspects
test outcomes to build or pick a rule, and never reads `ground_truth.json`. So a
candidate that overfits train but does not generalise loses to a shorter one —
the holdout + MDL, not the inducer's cleverness, is what makes the result honest.

## The four generative mechanisms (1–3 are built; 4 is later)

1. **Simplicity prior.** Candidates are emitted as a *nested* sequence of
   increasing length (0 clauses, 1, 2, …). The arbiter penalises program length
   on the holdout, so the **simplest adequate rule wins** — complexity is climbed
   only when the data demands it.
2. **Recombination.** The predicate pool is seeded with primitives exploded from
   the **library's** promoted rules (`--use-library`), so a discovery is reused,
   not re-derived. This is what makes the system accumulate.
3. **Surprise-directed search.** Each new clause is mined from the cases the
   current rule still gets wrong (the uncovered residuals), never from where it
   already predicts well.
4. **Analogy** (Mechanism 4) — importing structure across vocabularies — is the
   LLM's unique contribution and is left for last, expecting frailty (Level 5).

## The hypothesis class

A legible **decision list** (`DL_decision_list`): an ordered list of
`condition → outcome` clauses with a default. A condition is a **conjunction** of
mined primitives — `has(t)`, `[a->b] within lag`, `count(t)>=n`, `attr==value`.
Conjunctions plus ordering give negation for free, so the class expresses
hierarchies (clause 2 below means "P1 and not P2"):

```
IF [T1->F1] within 2 THEN reject ELIF [T2->F2] within 2 THEN reject
ELIF [T3->F3] within 2 THEN reject ELSE approve      # flat (levels 1–2)

IF has(T1) AND has(evt_Y) THEN reject ELIF has(T1) THEN review
ELIF count(evt_C)>=2 THEN review ELSE approve          # hierarchy (level 4)
```

That explicit, auditable rule **is** the deliverable.

## Two proposal strategies (the arbiter chooses)

The inducer emits candidates from two strategies and lets the arbiter pick on the
holdout — diverse proposals, MDL disposes:

- **A — gain-greedy single predicates:** minimal-length flat lists; recovers the
  flat levels (0–3) cheaply.
- **B — purity-grown conjunctions, purity-ordered:** seeds conjunctions from the
  top-gain predicates, grows each toward purity (preferring *general*, high-support
  refinements over coincidentally-pure ones), then lays them most-specific-first.
  This is what reaches hierarchies.

## Validation status (build order step 3)

Validated on Levels 1–2 by `tests/test_inducer.py`, and benchmarked across the
whole ladder with variance by `rule_induction/evaluate.py`:

```
level     recovery  halluc.  found    test_acc    bits_saved
level0      100%       —     100%    100.0%±0       155.5±2
level1      100%       —     100%    100.0%±0        79.7±3
level2      100%       —     100%    100.0%±0        78.3±6
level3      100%       —     100%    100.0%±0       155.2±8
level4        0%       —     100%     77.9%±2       106.9±7   ← partial (hierarchy hard)
level5        0%       —     100%     63.1%±8        20.6±4   ← analogy limit
neg           —         0%     0%       —             —       ← no false positives
```

This is the profile the plan predicts (Section 7): 0–3 solid, 4 partial, 5 fails,
and zero hallucination on the negative control. Level 4 finds real structure
(≈78% accuracy) but not the exact hierarchy — recovering the full table needs a
heavier learner (CN2-style lookahead or full library-driven composition).

## Run it

```bash
# Discover the rule behind a level's train split, validated on its holdout:
python -m rule_induction.inducer --data data --level level1 --seed 0

# Reuse promoted library primitives while searching (Mechanism 2):
python -m rule_induction.inducer --data data --level level2 --seed 0 --use-library

# Discover and promote the winner if it clears the library threshold:
python -m rule_induction.inducer --data data --level level1 --seed 0 --promote

# Benchmark the whole loop across levels x seeds (report the distribution):
python -m rule_induction.evaluate --seeds 5 -n 600 --sample-efficiency
```

Programmatic:

```python
from rule_induction.arbiter import load_split
from rule_induction.inducer import discover
from rule_induction.librarian import Librarian

train, test, _gt = load_split("data", "level1", 0)
result = discover(train, test, library=Librarian("library"),
                  promote=True, level_origin="level1")
if result["found_rule"]:
    print(result["best"]["hypothesis"]["description"])      # the legible rule
    print(result["best"]["verdict"]["bits_saved"])          # holdout compression
```

## Accumulation loop (how the three skills compose)

```
generator  ->  train/test holdout
inducer    ->  proposes candidate rules from train (M1–M3, reads library)
arbiter    ->  scores on holdout (MDL + sandbox), accepts the best
librarian  ->  stores the promoted rule; next induction recombines it (M2)
```

## Where things live

- **Method/judgment (prose):** this file.
- **Deterministic code:** `rule_induction/inducer.py` (mining + candidate
  generation + `discover` + CLI). It calls `arbiter.py` to judge and reads
  `librarian.py` for primitives.

Next: the **multi-agent layer** — only after Level 3 shows multimodal cases a
single investigator cannot separate. Add rival investigators with divergent
priors (one favors simplicity, one coverage, one distant analogies).
