"""Invented primitives — abstraction that earns its keep by reuse (Mechanism 4+).

The grammar in ``rules.py`` is a *fixed* vocabulary. The deepest thing humans do
is not recombine given concepts but **invent the primitives themselves** — carve a
new predicate out of the world when nothing existing compresses the data. This
module lets the agent do exactly that, under the same honesty discipline.

A **composed hypothesis** carries its own freshly-invented predicates plus a
decision list over them::

    {"kind": "composed",
     "name": "...",
     "primitives": {"count_gt": "def count_gt(events, a, b): ..."},
     "clauses": [{"all": [{"prim": "count_gt", "params": {"a": "up", "b": "down"}}],
                  "outcome": "flag"}],
     "default": "ok"}

Two-part MDL (the faithful part). The cost of a composed hypothesis is::

    L_program = sum over primitives of  (full code length   if NEW
                                         naming cost         if already in the library)
              + atoms(decision list) * BITS_PER_ATOM

Introducing a primitive pays its whole description length **once**; once it is in
the library it becomes shared vocabulary, so later hypotheses pay only a pointer.
That is why a good abstraction is *expensive to invent and cheap to reuse* — it
only becomes a bargain when it is reused, which is precisely how abstractions earn
their place. The arbiter still judges every composed hypothesis on the holdout;
invented code, being LLM-generated, runs in the **sandbox** (hostile-until-clean),
never in-process.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, FrozenSet, List

from .mdl import BITS_PER_ATOM, _count_atoms, program_bits_code

# Referencing a primitive already in the library costs a pointer into shared
# vocabulary, not its content — one atom, like naming a fixed grammar predicate.
NAMING_COST_BITS = BITS_PER_ATOM


def _pred_expr(ref: Dict[str, Any]) -> str:
    """A Python call expression for one predicate reference in a clause."""
    return f'{ref["prim"]}(events, **{ref.get("params", {})!r})'


def _clause_cond(clause: Dict[str, Any]) -> str:
    if "all" in clause:
        return "(" + " and ".join(_pred_expr(p) for p in clause["all"]) + ")"
    if "any" in clause:
        return "(" + " or ".join(_pred_expr(p) for p in clause["any"]) + ")"
    return _pred_expr(clause)   # single-predicate shorthand: {"prim": ..., "params": ...}


def compile_composed(hypothesis: Dict[str, Any]) -> str:
    """Stitch the invented predicates + decision list into one ``predict(events)``.

    The result is an ordinary code-hypothesis the existing sandbox can run.
    """
    lines: List[str] = []
    for src in hypothesis.get("primitives", {}).values():
        lines.append(textwrap.dedent(src).strip())
        lines.append("")
    lines.append("def predict(events):")
    for clause in hypothesis.get("clauses", []):
        lines.append(f"    if {_clause_cond(clause)}:")
        lines.append(f"        return {clause['outcome']!r}")
    lines.append(f"    return {hypothesis.get('default')!r}")
    return "\n".join(lines)


def program_bits_composed(hypothesis: Dict[str, Any],
                          known_primitives: FrozenSet[str] = frozenset()) -> float:
    """Two-part description length: full cost for new primitives, pointer for known."""
    bits = 0.0
    for name, src in hypothesis.get("primitives", {}).items():
        bits += NAMING_COST_BITS if name in known_primitives else program_bits_code(src)
    structure = {"clauses": hypothesis.get("clauses", []),
                 "default": hypothesis.get("default")}
    bits += _count_atoms(structure) * BITS_PER_ATOM
    return bits


def new_primitive_names(hypothesis: Dict[str, Any],
                        known_primitives: FrozenSet[str] = frozenset()) -> List[str]:
    """Primitive names this hypothesis introduces that the library does not yet hold."""
    return [n for n in hypothesis.get("primitives", {}) if n not in known_primitives]
