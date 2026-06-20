# Rule-Induction Agent — Project Plan & Synthetic Test Harness

> A system that reads sequences of events (groups of bits with causal structure) and
> recovers the **explicit, executable, auditable rules** that generated them — not just
> next-event prediction. Built as composable Claude Code skills over a persistent library
> of learned abstractions.

This document is self-contained. Hand it to a fresh Claude (ideally Claude Code) session
and continue building from any section.

---

## 1. What this system is (and is not)

**Is:** a rule *inducer*. Given event traces, it discovers the hidden generative rule, of
the form *"if A and then B within k steps, C tends to follow"*, and can lift many concrete
rules into one general rule with quantified variables (the "Galois moment" — seeing that
twenty concrete facts are shadows of one structure).

**Is not:** a next-token predictor, a search engine, or a RAG system. If the goal is
"predict the next event" use a sequence model. If the goal is "find/answer over documents"
use RAG. This system earns its place only when **all three** hold:

1. A compressible generative rule actually exists (not pure noise).
2. You want the rule *explicit and auditable*, not just a prediction.
3. The space of explanations is *multimodal* (rival theories compete) — this is what
   justifies the multi-agent layer; without it, a single generator suffices.

---

## 2. Core principle — the honesty anchor

The LLM **proposes**; deterministic code **verifies**. The single most important design rule:

> A rule is rewarded for **compression (MDL)**, not for accuracy.
> The most general rule is the one that most compresses the data — penalizing both error
> and rule complexity. This is what stops the system from "generalizing" into a giant rule
> that memorizes everything.

Two hard defenses, always on:

- **Holdout.** The inducer sees only train; the arbiter judges on held-out test data. A
  rule that only "works" on seen data is memorization in disguise.
- **MDL cost includes program length.** A function that memorizes the trace is enormous,
  so its description cost is high and it loses to any short rule explaining the same data.

When the inducer is allowed to emit *code* as a hypothesis, add a third defense:

- **Sandbox.** Treat every code-hypothesis as hostile until a sandboxed run returns clean:
  timeout, no network, no filesystem, memory cap.

---

## 3. The four generative mechanisms (where hypotheses come from)

The hard part is not verification — it's **generating** the few hypotheses worth testing
out of an infinite space. Modeled on how cognition does it:

1. **Simplicity prior (Mechanism 1).** Don't sample hypotheses uniformly — bias generation
   toward short programs. Start simple, climb in complexity only when simple fails. With
   code-hypotheses this stops being an optimization and becomes the *only* thing that keeps
   search tractable.
2. **Recombination (Mechanism 2).** New hypotheses are new configurations of old
   primitives, not symbols from nothing. This is why the shared library matters: every
   accepted abstraction enlarges the lego set for the next hypothesis.
3. **Surprise-directed search (Mechanism 3).** Generate hypotheses *where the current best
   rule's prediction fails* — the residuals. Don't spend generation budget where you
   already predict well.
4. **Analogy (Mechanism 4).** Import structure from another domain ("this looks like a
   queue / a contagion / a counter mod k"). This is the LLM's *unique* contribution and the
   one a symbolic solver cannot do. It is also the most fragile — expect transfer across
   distant domains to be weak. That's the honest state-of-the-art limit, not a bug.

---

## 4. Architecture — skills + persistent store

Decompose by **task the user recognizes**, not by internal component. Three skills plus a
persistent store (the "librarian", which is *state*, not a skill).

### Skill 1 — Inducer (the orchestrator)
- **Triggers when:** there is an event trace / sequence and the user wants to extract
  rules, patterns, or causal structure.
- **Does:** encoding → base mining → hypothesis generation (Mechanisms 1–4) → calls the
  arbiter → assembles the rule hierarchy. Reads the library to get primitives to recombine.

### Skill 2 — Arbiter (the honesty anchor)
- **Triggers when:** there is a candidate rule/hypothesis to validate against data.
- **Does:** train/test split → execute hypothesis in **sandbox** → measure MDL compression
  on **holdout** → accept/reject. On acceptance above the *library threshold*, **promotes**
  the rule to the store.
- **Ships real code** (not prose): an MDL scorer + a sandbox runner. These must be
  deterministic and repeatable.

### Skill 3 — Synthetic data generator (the test bench)
- **Triggers when:** you need to test the inducer or validate that it recovers known rules.
- **Does:** generates the difficulty ladder (Section 6) with the planted ground-truth rule
  recorded, plus the negative control.

### The Librarian — persistent store (NOT a skill)
- A **git-versioned directory** of accepted abstractions. Each entry: the
  function/rule + its MDL score + metadata (what it compresses, when accepted).
- **Written by** the arbiter (on promotion). **Read by** the inducer (at generation time).
- This is what makes the system *accumulate*: each validated discovery becomes a primitive
  for the next, enabling abstractions-of-abstractions. Git gives auditability and lets you
  revert if a false abstraction contaminates the library.
- **Stricter threshold to enter the library than to be accepted in one run** — admitting a
  rule as a permanent foundation is a bigger commitment than using it once.

### What goes where
- **SKILL.md (prose):** method and judgment — the MDL principle, why holdout is mandatory,
  the ladder, sandbox warnings, when to climb in complexity.
- **Scripts (code):** anything that must be deterministic and repeatable — the MDL scorer,
  the sandbox runner, the level generators.

---

## 5. The multi-agent layer (only if Level 3 justifies it)

Do **not** map one agent per component (inducer-agent, arbiter-agent…) — that's a pipeline
in costume. The real value of multi-agent is **parallel search of incompatible theories
under competitive pressure**. Decompose by **rival theory**, not by function:

- **Multiple Investigators**, each a *complete* generate-and-verify loop, but with
  *deliberately different priors* (one favors simplicity, one favors coverage, one chases
  distant analogies). Diversity is forced by bias, not by chance.
- **Arbiter / Reviewer** runs rival theories on a **fresh holdout**, measures real
  compression (not self-reported), and reports where each theory breaks — that becomes the
  next agenda.
- **Librarian** = shared memory; a discovery by Investigator A becomes a primitive B can
  recombine. Collective capacity grows faster than the sum of individuals.

**Warnings:** coordination cost is real — for a single clear rule this is overkill. And
LLM-only "debate" degenerates (mutual flattery or unresolved divergence); the deterministic
arbiter on holdout is the *only* thing preventing both collapses. Don't trust agents to
convince each other; trust held-out data to refute them.

---

## 6. Synthetic dataset — the difficulty ladder

Generate **a ladder of generators**, not one dataset. Each level isolates one capability,
so a failure tells you *exactly* what's missing. Domain below is **claims/transactions
case-files** (causal process, FCRM-aligned, the explicit rule is the deliverable).

### Event / case-file format

```json
{
  "case_id": "c_00001",
  "events": [
    {"t": 0, "type": "open",        "attrs": {"channel": "web"}},
    {"t": 1, "type": "evt_A",       "attrs": {"amount": 120}},
    {"t": 2, "type": "evt_B",       "attrs": {"region": "north"}},
    {"t": 5, "type": "evt_C",       "attrs": {}}
  ],
  "outcome": "reject",
  "level": 1,
  "planted_rule_id": "R1_typed_successor"
}
```

- `events` is an **ordered** sequence (the trace). Each event has a discrete `type` and an
  optional `attrs` bag (this is the "group of bits" — several flags/values at once).
- `outcome` is decided by the **planted rule** for that level.
- `planted_rule_id` is the **ground truth** the system must recover. Never shown to the
  inducer — only to the scorer.

### The levels

| Level | Name | Planted rule (ground truth) | Capability tested |
|------|------|------------------------------|-------------------|
| 0 | Sanity | If `evt_X` appears → outcome always `reject` | basic recovery; if this fails it's a bug |
| 1 | Recombination | `evt_type(k)` at lag → `outcome_type(k+1)`, instantiated over several k | abstraction: lift concrete → general (the Galois test) |
| 2 | Noise | Level-1 rule + random distractor events injected per case | recovery under dirty signal |
| 3 | Multimodal | Two distinct rules govern disjoint case subsets (e.g. by `channel`) | **justifies multi-agent**: rival theories, arbiter separates them |
| 4 | Compositional | Rule depends on a rule (2–3 level hierarchy) | librarian composes abstractions of abstractions |
| 5 | Analogy | Same abstract structure in two different vocabularies | Mechanism 4 (expect frailty — honest limit) |
| NEG | Negative control | **No rule** — pure noise outcomes | false-positive rate; must extract ~nothing "confident" |

### Mandatory controls

- **Holdout per level:** split case-files train/test; inducer sees train, arbiter scores on
  test.
- **Negative control (NEG):** run the whole system on rule-free data. If it returns
  "confident" rules, that's your baseline false-positive rate — the most revealing and most
  skipped check.
- **Multiple seeds + variance:** run each level over many generator seeds; report the
  distribution, not the best case. (Same discipline as eval/benchmark variance analysis.)

---

## 7. Metrics (what synthetic data uniquely enables)

Because the planted rule is known, measure against ground truth — impossible on real data:

- **Rule recovery:** did the system produce a rule *logically equivalent* to the planted
  one (even if written differently)?
- **Hallucination rate:** how many accepted rules are *not* in the generator? Most
  important, and unmeasurable on real data.
- **Abstraction level reached:** stuck at concrete rules, or lifted to the general one?
- **Sample efficiency:** how many case-files needed to recover the rule? Good inducers need
  few.

### Expected results (so you don't miscalibrate)
- Levels 0–2: should be solid if skills are well built.
- Level 3: this is where you *see whether multi-agent earns its cost*. If multi-agent
  doesn't beat a single generator here, drop back to one generator.
- Level 4: expect irregular.
- Level 5: expect failure or fragility — known SOTA limit. A system that nails 0–3, does 4
  partially, and fails 5 is a **good** result. Distrust anyone claiming consistent Level 5.

---

## 8. Build order (recommended)

1. **Skill 3 — level generators (0–3) + NEG.** Gives you the bench to test everything else.
   Record ground truth per case.
2. **Skill 2 — arbiter** (MDL scorer + sandbox + holdout). The anchor: it both *judges* and
   *writes to the library*, so its integrity defines everything downstream. Validate it on
   Levels 0–1: it must accept the true rule and reject memorizers + resist the NEG bait.
3. **Skill 1 — inducer** (single-investigator first; Mechanisms 1–3). Validate on
   Levels 1–2.
4. **Librarian** wiring: arbiter writes on promotion, inducer reads at generation. Validate
   accumulation on Level 4.
5. **Multi-agent layer** — *only after* Level 3 shows multimodal cases a single generator
   can't separate. Add rival investigators with divergent priors.
6. **Analogy (Mechanism 4)** — last, expecting frailty. Test on Level 5.

---

## 9. Corpus / document extension (optional)

If the input is a **document corpus** rather than raw traces, RAG is the *sensory organ*,
not a competitor: use RAG (hybrid BM25+vector + knowledge graph) to parse each document into
a structured event trace (facts with entity + timestamp), then run the inducer over the
extracted traces. RAG reads at scale; the inducer discovers the causal pattern spread across
thousands of documents that no single reader sees.

**Filter question:** is there a hidden *process with rules* generating these documents whose
rules you'd want to recover (claims, transactions, case-files, tickets, histories)? If yes,
this system fits. If the documents are ends in themselves (essays, articles, topical text),
stay with RAG — the inducer would spin in a vacuum.

---

## 10. Who this is useful for (honest scoping)

The user is whoever satisfies the three conditions in Section 1 — narrow but defensible:

- **Compliance / financial-crime / audit:** need *auditable* rules, not opaque scores;
  explainability is a regulatory *requirement*, so "legible rule" is the deliverable, not a
  luxury. Strongest fit.
- **Reverse engineering of opaque systems:** protocols, proprietary formats, distributed-log
  causal rules (SRE/observability — "when service X emits this, that fails k later").
- **Researchers with process data** looking for the governing law (more fragmented as a
  market; open-source/teaching fit).
- **Pedagogical (most solid today):** built as composable skills, this is a near-perfect
  case study of harness engineering — generator, verifier, persistent memory, multi-agent
  decomposition, and the honesty anchor (LLM proposes / code verifies) all at once.

**Honest caveat:** most people with "a big corpus" are *not* users — they want search/Q&A
(RAG). This serves the narrow band with a hidden rule-governed process needing auditable
recovery. Frame it as: solves a concrete FCRM problem, teaches the course, and demonstrates
capability — any compliance customer that emerges is upside, not the thesis.

---

## 11. Suggested first prompt for the next session

> "I'm building a rule-induction agent (full plan attached as markdown). Start with Skill 3:
> generate the synthetic claims/case-file dataset for Levels 0–3 plus the negative control,
> in the JSON format in Section 6, with planted ground-truth rules recorded separately from
> the case files. Then we'll build the arbiter (Skill 2) and validate it against these
> levels. Use Python, no heavy frameworks."
