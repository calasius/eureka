---
name: investigator
description: >-
  Act as a rival investigator (or a panel of them) to discover the rule behind an
  event trace when the deterministic inducer is not enough — multimodal cases,
  hierarchies, or cross-vocabulary analogies. YOU (the Claude agent) are the
  proposer: you read the train traces, propose explicit hypotheses with a
  deliberate prior, and hand each to the arbiter, which verifies on a holdout. Use
  when a level needs creative or analogical hypotheses a symbolic search can't reach.
---

# Investigator — the multi-agent layer (Section 5)

This is where the plan's "LLM proposes / code verifies" becomes literal: **the LLM
is you**, the running Claude agent. There is no API call and no model running
inside a Python process — you read the data through a skill and propose, and the
deterministic arbiter (MDL + sandbox, on a fresh holdout) is the only thing that
decides what's true. Investigators never convince each other; held-out data
refutes them.

Use this layer only when a single deterministic investigator (the `inducer`
skill) is not enough: multimodal cases where rival theories compete, hierarchies,
or — most importantly — **analogy across vocabularies (Mechanism 4)**, which a
symbolic solver cannot do and you uniquely can.

## Entry point & arguments

Invoke this skill by name in a Claude Code session; the rest of the line is what
to analyze. The agent (you) then reads it and runs the loop below.

```
/investigator level4                 # a bench level (default seed 0)
/investigator level4 seed 2          # a bench level, specific seed
/investigator --dataset path/to/dir  # YOUR own data: a folder with train.jsonl + test.jsonl
```

Headless (one shot from the shell):

```bash
claude -p "/investigator level5 — recover the rule and promote the winner"
claude -p "/investigator --dataset /data/claims — find the rule behind these traces"
```

**What you pass** is either a bench `level [seed]`, or `--dataset <dir>` pointing
at a folder that holds your own `train.jsonl` and `test.jsonl` (one case per line,
in the Section-6 format; `ground_truth.json` optional). The reading and verifying
tools take the same argument:

```bash
python -m rule_induction.present  --dataset <dir>                 # train-only view
python -m rule_induction.arbiter  --dataset <dir> --hypothesis hyp.json --promote
```

The agent never runs the *proposing* itself as a command — proposing is you,
reasoning over the train view. `present`/`arbiter`/`librarian` are your instruments.

## The honesty boundary (non-negotiable)

You may read **train only**. Never open `test.jsonl` outcomes or
`ground_truth.json` to form or pick a hypothesis — that is the answer, and the
whole point is that the arbiter judges you blind on the holdout. The view tool
enforces this: it reads only `train.jsonl`.

```bash
python -m rule_induction.present --data data --level level5 --seed 0
```

## The loop

1. **Read** the train view above: the traces, the event vocabulary, the
   attributes, the label distribution.
2. **Propose** a hypothesis with a deliberate prior (see below). Write it as JSON:
   - a symbolic **rule** using the grammar
     (`marker`, `typed_successor`, `count_at_least`, `attr_eq`, `DL_decision_list`), or
   - a **code** hypothesis: `{"kind":"code","source":"def predict(events): ..."}`.
     Code runs in the sandbox (hostile-until-clean); use it for structure the
     grammar can't express (e.g. counting mod k, dispatch on a per-case vocab).
3. **Verify** on the holdout — the arbiter, not you, decides:
   ```bash
   python -m rule_induction.arbiter --data data --level level5 --seed 0 --hypothesis hyp.json --promote
   ```
4. **Chase residuals** (Mechanism 3): if accepted-but-imperfect, ask the view for
   the train cases your rule still gets wrong and propose a refinement targeting
   exactly those:
   ```bash
   python -m rule_induction.present --data data --level level5 --seed 0 --residuals hyp.json
   ```
5. **Accumulate**: a promoted rule enters the librarian and becomes a primitive
   the next investigation can recombine (`--use-library` on the inducer).

## Rival investigators = divergent priors

Diversity is forced by **bias, not chance**. Run several investigators, each
committed to a different prior, and let the arbiter separate them on the holdout:

| Investigator | Prior | Proposes |
|---|---|---|
| **Simplicity** | shortest program first | one marker / one successor clause; climbs only if it fails |
| **Coverage** | explain the most cases first | a high-coverage rule, then refine its residuals |
| **Analogy** | "what known structure is this?" | imports a template — counter mod k, queue, contagion, parity — usually as a `code` hypothesis. **This is the one a symbolic solver can't do.** |

Two ways to run the panel, both with you in every agent and no API call:

- **Inline**: act as each investigator yourself, in turn — propose under
  simplicity, then coverage, then analogy — collecting each verdict.
- **Parallel subagents**: spawn rival investigators with the **Agent tool**, each
  a Claude agent handed the train view and one prior, each returning a hypothesis
  JSON. You (the orchestrator) then run every proposal through the arbiter, pick
  the highest `bits_saved` on the holdout, and promote it. The librarian is the
  shared memory across them.

A **reviewer** pass (also you) runs the surviving theories on the holdout, reports
where each breaks (its residuals), and that becomes the next agenda.

## When this earns its cost

For a single clear rule, this is overkill — use the `inducer` skill. The
multi-agent layer pays off only when theories genuinely compete or when the rule
needs an analogical leap. The honest target (Section 7): the deterministic
inducer nails levels 0–3; this layer is what reaches **level 5 (analogy)**, the
known SOTA limit, by proposing a counter-mod-k code hypothesis the symbolic
search never finds.

## Where things live

- **Method/judgment (prose):** this file — you follow it.
- **Reading surface:** `rule_induction/present.py` (train-only view + residuals).
- **Verifier:** `rule_induction/arbiter.py` (MDL + sandbox on holdout) — unchanged;
  it judges your proposals exactly as it judges the inducer's.
- **Shared memory:** `rule_induction/librarian.py` (the git-versioned store).
