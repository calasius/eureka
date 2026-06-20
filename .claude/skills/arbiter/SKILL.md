---
name: arbiter
description: >-
  Validate a candidate rule/hypothesis against data — the honesty anchor. Splits
  train/test, executes code-hypotheses in a sandbox, measures MDL compression on
  the holdout, and accepts/rejects. On acceptance above the library threshold it
  promotes the rule to the persistent store. Use whenever there is a candidate
  rule to judge.
---

# Arbiter — the honesty anchor (Skill 2)

The LLM **proposes**; this skill (deterministic code) **verifies**. It is the
integrity boundary of the whole system: it both *judges* hypotheses and *writes
to the library*, so everything downstream inherits its honesty.

## The one principle

> A rule is rewarded for **compression (MDL) on holdout**, not for accuracy.

`bits_saved = L_null − L_program − L_data|H`, all measured on **test** data the
inducer never saw, with a *prequential plug-in code* (the coding distribution is
learned on train, the bits are paid on test). See `rule_induction/mdl.py`.

## Three hard defenses (always on)

1. **Holdout.** The inducer sees only `train.jsonl`; the arbiter scores on
   `test.jsonl`. A rule that only "works" on seen data is memorization — it buys
   no bits here, because the code is learned on train and the cost is paid on
   unseen test.
2. **Program length in the MDL cost.** A function that memorizes the trace is
   enormous, so `L_program` is huge and it loses to any short rule explaining the
   same data. (`mdl.program_bits_spec` for symbolic rules; `program_bits_code`,
   a token count, for code.)
3. **Sandbox.** Every *code*-hypothesis is hostile until a sandboxed run returns
   clean: wall-clock timeout, CPU cap, memory cap, no filesystem writes,
   restricted builtins, isolated interpreter. Any dirty run is an automatic
   rejection. See `rule_induction/sandbox.py`. (Symbolic `rule` hypotheses are
   built from our own registry and run in-process — trusted.)

   *Honest limit:* full network isolation needs OS-level confinement
   (nsjail/firejail/container). The restricted builtins stop the candidate from
   importing `socket`, but run genuinely untrusted code under a container.

## Two thresholds (Section 4)

- **Run-acceptance** (lower, default 2.0 bits): enough to *use* a rule once.
- **Library threshold** (stricter, default 8.0 bits, in `library/config.json`):
  to admit a rule as a *permanent foundation*. Enforced by the Librarian on
  promotion. Accepting in a run does not guarantee promotion.

## Validation status (build order step 2)

Validated on Levels 0–1 by `tests/test_arbiter.py`: it **accepts** the true rule
(large positive `bits_saved`, test accuracy ≈ 1.0), **rejects** a memorizer
(huge program + holdout failure ⇒ negative `bits_saved`), and **resists the NEG
bait** (no rule compresses rule-free data ⇒ rejection).

## Run it

A hypothesis is JSON. Symbolic rule:

```json
{"kind": "rule", "rule_id": "R0_marker_presence",
 "params": {"marker": "evt_X", "hit": "reject", "miss": "approve"},
 "name": "marker evt_X", "description": "evt_X present -> reject"}
```

Code hypothesis (runs in the sandbox):

```json
{"kind": "code",
 "source": "def predict(events):\n    return 'reject' if any(e['type']=='evt_X' for e in events) else 'approve'",
 "name": "marker evt_X (code)", "description": "..."}
```

```bash
# Judge against a level's holdout:
python -m rule_induction.arbiter --data data --level level0 --seed 0 --hypothesis hyp.json

# Judge and promote to the library if it clears the library threshold:
python -m rule_induction.arbiter --data data --level level1 --seed 0 --hypothesis hyp.json --promote
```

Programmatic:

```python
from rule_induction.arbiter import load_split, adjudicate
from rule_induction.librarian import Librarian

train, test, _gt = load_split("data", "level1", 0)   # arbiter never uses _gt to predict
verdict = adjudicate(hyp, train, test, librarian=Librarian("library"), level_origin="level1")
```

## Boundary it must not cross

The arbiter judges hypotheses against held-out **data**, never against
`ground_truth.json`. The planted rule is for the offline metrics
(`rule_induction/metrics.py`: rule recovery, hallucination), not for adjudication.

## Where things live

- **Method/judgment (prose):** this file.
- **Deterministic code:** `mdl.py` (scorer), `sandbox.py` (runner),
  `arbiter.py` (orchestration + CLI), `metrics.py` (ground-truth checks).

Next in the build order: the **inducer** (Skill 1) — it generates hypotheses
(Mechanisms 1–3), hands them here, and reads promoted primitives from the library.
