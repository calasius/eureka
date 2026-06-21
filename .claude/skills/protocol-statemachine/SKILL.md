---
name: protocol-statemachine
description: >-
  Given traces of a protocol (sequences of messages, all valid), recover the finite
  STATE MACHINE that generated them and return a Markdown report with a Mermaid
  diagram. Unsupervised — learns from positive traces only, no labels and no
  fabricated invalid sequences. Deterministic Python infers the structure (states +
  transitions by k-tails state-merging); YOU (the agent) give states semantic names
  and write the explanation. Use for TCP/TLS/SIP/HTTP handshakes, API call
  sequences, OpenTelemetry traces — anything where you want the protocol's automaton.
---

# Protocol state-machine inference (unsupervised)

This is the **unsupervised** member of Eureka: there is no `outcome` label to
predict and no need to fabricate invalid traces. The whole sequence *is* the
object of interest, and the state machine is the abstraction that compresses it.
The honesty thread still holds — the machine is kept only because it **compresses**
the traces (vs. a structureless baseline) and **generalises** (accepts traces with
more loop iterations than were ever seen).

## Division of labour

- **Deterministic engine** (`rule_induction/protocol.py`): builds the prefix-tree
  acceptor, merges states by **k-tails**, determinises to a clean DFA, and renders
  Markdown + a Mermaid `stateDiagram-v2`. This is exact and not your job to redo.
- **You (the agent):** choose the symbol mapping (which event field is the message,
  whether to fold an attribute like `status`), pick a good `k`, then **rename the
  states with protocol semantics** (`CLOSED`, `SYN_SENT`, `ESTABLISHED`, `CLOSING`…)
  and write the prose that explains the machine. The engine emits `S0..Sn`; you make
  them meaningful.

## The loop

1. **Get the structure.** Run the engine on the traces:
   ```bash
   python -m rule_induction.protocol --dataset <dir>            # or --demo
   python -m rule_induction.protocol --dataset <dir> --attr status --k 2 --out sm.md
   ```
   It prints (or writes) the Markdown: overview, Mermaid diagram, transition table,
   the compression saving, and any loops it generalised.
2. **Sanity-check with the scratchpad** if unsure about the alphabet — e.g. which
   event types actually occur, how long traces run (`rule_induction.explore`).
3. **Rename + explain.** Map `S0..Sn` to protocol states from the transition
   structure (the start state, the accepting "clean end", the looped state =
   data-transfer/keep-alive). Rewrite the Mermaid labels and write a short
   paragraph per state: what it means and which messages leave it.
4. **Report.** Present the final Markdown — semantic Mermaid diagram + explanation —
   noting the compression and the generalisation (it accepts unseen loop counts).

## Input format

Traces are cases in the usual event format (`{events:[{type, attrs}], ...}`); the
engine reads `train.jsonl`(+`test.jsonl`) from the folder and uses each event's
`type` as the message symbol (`open`/`close` are dropped). Real captures convert
cleanly: one session per case (group by `tcp.stream`, `Call-ID`, `trace.id`…), one
message per event, drop high-cardinality fields (IPs, ports, exact timestamps).

## When this earns its place

Use it whenever the question is *"what protocol/automaton generated these
sequences?"* and you only have **valid** traces — the supervised rule-induction
skills need an outcome and would force you to fabricate negatives; this does not.
For the literal minimal automaton with guarantees, dedicated tools (flexfringe,
AALpy/LearnLib) go further; this gives a legible, compressing, generalising machine
plus a human explanation, in one pass.
