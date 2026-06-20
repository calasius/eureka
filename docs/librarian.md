# The Librarian — persistent store (NOT a skill)

The Librarian is the system's memory: a **git-versioned directory of accepted
abstractions** (Section 4 of the plan). It is *state*, not a skill — it does not
"trigger." It is **written by the arbiter** (on promotion) and **read by the
inducer** (at generation time, to get primitives to recombine — Mechanism 2).
This is what lets the system *accumulate*: each validated discovery becomes a
primitive for the next, enabling abstractions-of-abstractions.

## Why git

Git gives two things the plan demands:
- **Auditability** — every promotion/revert is a commit with a message recording
  what was admitted, its compression gain, and where it came from.
- **Revert** — if a false abstraction ever contaminates the library, you can
  remove it cleanly and the history shows it happened.

## Two thresholds (and which one this enforces)

- **Run-acceptance threshold** — used once, by the arbiter, to accept a rule in a
  single run. Enforced by the arbiter (not yet built).
- **Library threshold (STRICTER)** — to admit a rule as a *permanent foundation*.
  Enforced here, in `Librarian.promote`, against `entry.mdl.bits_saved` (the
  compression gain measured **on holdout**). Admitting a reusable primitive is a
  bigger commitment than using a rule once, so the bar is higher.

Defaults live in `library/config.json`
(`run_acceptance_threshold_bits: 1.0`, `library_threshold_bits: 8.0`).

## Layout

```
library/
  config.json          # thresholds + metadata
  index.json           # summary of every entry (fast read for the inducer)
  entries/<id>.json     # one abstraction per file (one fact per file)
  README.md
  (.git — every promote/revert is a commit)
```

An entry records: `name`, `kind` (`rule`|`function`), `description`, `spec`
(rule params) and/or `program` (code), `rule_id` (link to a known rule family),
`mdl` (`bits_saved`, `compresses`, `score`), and `provenance` (`run_id`,
`investigator`, `level_origin`, `evaluated_on`).

## API (what the arbiter and inducer call)

```python
from rule_induction.librarian import Librarian, PromotionRejected

lib = Librarian("library")                      # init repo on first use

# --- arbiter, on acceptance above the library threshold ---
entry = {
    "name": "typed-successor T1->F1 (lag 2)",
    "kind": "rule",
    "description": "Tk followed by Fk within 2 steps -> reject.",
    "rule_id": "R1_typed_successor",
    "spec": {"pairs": [["T1", "F1"]], "lag": 2, "hit": "reject", "miss": "approve"},
    "mdl": {"bits_saved": 42.0, "compresses": "level1 outcomes"},
    "provenance": {"run_id": "run_007", "investigator": "simplicity", "level_origin": "level1"},
}
record = lib.promote(entry)                     # git-committed; raises PromotionRejected if too weak

# --- inducer, at generation time ---
for prim in lib.list(kind="rule"):              # primitives to recombine
    full = lib.get(prim["id"])

# --- audit / cleanup ---
lib.revert("typed-successor-t1-f1-lag-2", reason="false positive on NEG re-check")
print(lib.log())
```

## CLI

```bash
python -m rule_induction.librarian list
python -m rule_induction.librarian show <id>
python -m rule_induction.librarian promote --file entry.json [--bits 42] [--threshold 8]
python -m rule_induction.librarian revert <id> --reason "..."
python -m rule_induction.librarian log
```

## Notes

- `promote` is **idempotent-ish**: re-promoting an existing id only creates a new
  version if it strictly improves the recorded compression; otherwise it's
  rejected (no churn).
- The store works without git (commits become no-ops) but you lose the audit
  trail and revert guarantee — keep git on in real use.
- `library/` is its own git repo, independent of the project repo, so the
  abstraction history is self-contained and portable.
