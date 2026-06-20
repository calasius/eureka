#!/usr/bin/env bash
# Eureka — single entry point to run the rule-induction system end to end.
#
# Two layers:
#   * deterministic (generator, inducer, arbiter, librarian, evaluate) — pure shell.
#   * the agent (the `investigator` skill) — the Claude agent proposes hypotheses;
#     the deterministic arbiter verifies them. Invoked via `claude -p "/investigator ..."`.
#
# Usage:
#   ./run.sh bench                 generate the ladder + deterministic profile (0-5 + neg)
#   ./run.sh discover <level>      run the deterministic inducer on a level, promote the winner
#   ./run.sh investigate <target>  run the AGENT (investigator skill) on a level or your dataset
#   ./run.sh library               show the accumulated abstraction library + git audit trail
#   ./run.sh full                  the whole thing: bench -> investigate level4 & level5 -> library
#   ./run.sh test                  run the test suite
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

bench() {
  echo "== 1. Generate the difficulty ladder (levels 0-5 + negative control) =="
  "$PY" -m rule_induction.generate --out data --seeds 3 -n 600
  echo
  echo "== 2. Deterministic profile across levels x seeds (Section 7 metrics) =="
  "$PY" -m rule_induction.evaluate --seeds 3 -n 600
}

discover() {
  local level="${1:?usage: ./run.sh discover <level>}"
  echo "== Deterministic inducer on ${level} (propose -> arbiter -> promote) =="
  "$PY" -m rule_induction.inducer --data data --level "$level" --seed 0 --use-library --promote
}

investigate() {
  local target="${1:?usage: ./run.sh investigate <level | --dataset DIR>}"
  shift || true
  local prompt="/investigator ${target} $* — read the train view, propose hypotheses (rival priors, analogy if needed), verify each with the arbiter on the holdout, and promote the winner."
  if command -v claude >/dev/null 2>&1; then
    echo "== Agent (investigator skill) on ${target} =="
    claude -p "$prompt"
  else
    echo "The 'claude' CLI is not on PATH. Run the agent step inside a Claude Code session:"
    echo
    echo "    claude"
    echo "    > /investigator ${target} ${*}"
    echo
    echo "The agent will use these instruments itself (train-only view, then verify):"
    echo "    $PY -m rule_induction.present  ${target/#level/--level } --limit 30"
    echo "    $PY -m rule_induction.arbiter  --hypothesis hyp.json --promote   # +the same target"
  fi
}

library() {
  echo "== Accumulated abstraction library =="
  "$PY" -m rule_induction.librarian list
  echo
  echo "== Git audit trail (every promotion / revert) =="
  "$PY" -m rule_induction.librarian log | head -8
}

full() {
  bench
  echo
  echo "== 3. Agent layer: rival investigators on the hard levels (4 hierarchy, 5 analogy) =="
  investigate level4 --promote
  investigate level5 --promote
  echo
  library
}

cmd="${1:-full}"; shift || true
case "$cmd" in
  bench)       bench ;;
  discover)    discover "$@" ;;
  investigate) investigate "$@" ;;
  library)     library ;;
  full)        full ;;
  test)        "$PY" -m unittest discover -s tests ;;
  *) echo "unknown command: $cmd"; sed -n '8,16p' "$0"; exit 1 ;;
esac
