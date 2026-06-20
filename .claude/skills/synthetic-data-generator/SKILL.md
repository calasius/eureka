---
name: synthetic-data-generator
description: >-
  Generate the synthetic claims/case-file test bench for the rule-induction
  agent — the difficulty ladder (levels 0–5 + negative control) with planted
  ground-truth rules recorded separately from the traces. Use this when you need
  to test the inducer/arbiter or validate that the system recovers known rules.
---

# Synthetic Data Generator — the test bench (Skill 3)

This is the **bench**, built first (Section 8 of the plan): generate it before
anything else so the arbiter and inducer have something to be validated against.
It produces a *ladder of generators*, not one dataset — each level isolates one
capability, so a failure tells you exactly what is missing.

## Method & judgment (read this before running)

- **Ground truth is planted and recorded separately.** Each level has a planted
  rule (`planted_rule_id` + params). Outcomes are produced by that exact rule, so
  the scorer can later check a recovered rule for *logical equivalence*. The
  planted rule lives in `ground_truth.json` — **never feed it to the inducer.**
  Only the scorer/arbiter may read it.
- **Holdout is mandatory.** Every (level, seed) is split train/test. The inducer
  sees `train.jsonl`; the arbiter scores on `test.jsonl`. A rule that only works
  on train is memorization in disguise.
- **The negative control is the most revealing and most skipped check.** Level
  `neg` has *no rule* — outcomes are i.i.d. coin flips, independent of the events
  (rule-looking bait events are present on purpose). Run the whole system on it:
  anything it reports "confident" there is your false-positive baseline.
- **Multiple seeds, report the distribution.** Generate several seeds per level
  and report variance, never the best case.
- **Clean separation keeps ground truth exact.** Neutral filler event types are
  disjoint from the types any rule reacts to, so neutrals can never accidentally
  fire a rule. Levels plant *hard negatives* (trigger-without-confirm,
  confirm-too-far) so a level distinguishes real structure from cheap proxies
  like mere presence.

## The ladder (what each level tests)

| Level | Planted rule | Capability under test |
|-------|--------------|-----------------------|
| `level0` | marker `evt_X` present → reject | basic recovery (if this fails, it's a bug) |
| `level1` | `Tk` followed by `Fk` within lag 2, over 3 pairs | lift concrete → general (the Galois test) |
| `level2` | level-1 rule + injected distractor events | recovery under dirty signal |
| `level3` | two rules on disjoint subsets by `channel` | **justifies multi-agent**: rival theories |
| `level4` | outcome depends on sub-rules (P1,P2,P3 hierarchy) | composing abstractions of abstractions |
| `level5` | count(trigger) mod 3, in two vocabularies | analogical transfer (expect frailty — SOTA limit) |
| `neg` | **no rule** (i.i.d. outcomes) | false-positive rate |

Expected results (so you don't miscalibrate): 0–2 solid; 3 is where multi-agent
must earn its cost; 4 irregular; 5 expect failure/fragility. Nailing 0–3, partial
4, failing 5 is a **good** result.

## Run it

Python, stdlib only — no install needed. From the repo root:

```bash
# Full ladder, 3 seeds each, 200 cases per (level, seed):
python -m rule_induction.generate --out data

# Build-order starter set (levels 0–3 + NEG), more seeds:
python -m rule_induction.generate --out data --levels level0 level1 level2 level3 neg --seeds 5

# Bigger, custom train ratio:
python -m rule_induction.generate --out data -n 500 --seeds 10 --ratio 0.6
```

Output layout:

```
data/<level>/seed_<s>/train.jsonl      # inducer input — NO planted_rule_id
                      test.jsonl        # arbiter holdout
                      ground_truth.json # scorer-only: planted rule + params + label distribution
data/manifest.json                      # index of everything generated
```

Each line of `train.jsonl` / `test.jsonl` is one case-file:

```json
{"case_id": "c_00001",
 "events": [{"t": 0, "type": "open", "attrs": {"channel": "web"}},
            {"t": 1, "type": "evt_X", "attrs": {}}],
 "outcome": "reject", "level": "level0", "split": "train"}
```

## Programmatic use

```python
from rule_induction.dataset import build, generate_level
from rule_induction.rules import make_labeler

cases, gt = generate_level("level1", n=200, seed=0)
# Re-instantiate the exact planted rule (what the scorer uses):
labeler = make_labeler(gt["planted_rule_id"], gt["params"])
assert all(labeler(c["events"]) == c["outcome"] for c in cases)  # ground truth is exact
```

## Where things live

- **Method/judgment (prose):** this file.
- **Deterministic, repeatable code:** `rule_induction/levels.py` (level generators),
  `rule_induction/rules.py` (the planted labelers / ground-truth registry),
  `rule_induction/dataset.py` (holdout split + on-disk layout),
  `rule_induction/generate.py` (CLI).

Next in the build order: the **arbiter** (Skill 2 — MDL scorer + sandbox +
holdout), validated on levels 0–1; then the **inducer** (Skill 1). The
**Librarian** (persistent store) is already wired — see `docs/librarian.md`.
