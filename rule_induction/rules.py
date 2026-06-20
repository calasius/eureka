"""Planted rules — the ground truth labelers (Section 6).

This module is the *single source of truth* for outcomes: every level generator
labels its cases by calling the same ``make_labeler`` the scorer will use to
check recovery. A labeler maps an ordered list of events -> outcome string.

Ground truth is stored as ``(planted_rule_id, params)`` — pure JSON — and
``make_labeler`` reconstitutes the executable rule from it. This keeps the
arbiter/scorer fully reproducible: it can re-instantiate the exact rule that
generated any level and test a candidate for *logical equivalence* against it.

The ``planted_rule_id`` is NEVER shown to the inducer; only the scorer sees it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .model import Event, open_attr

Labeler = Callable[[List[Event]], str]
Predicate = Callable[[List[Event]], bool]


# --------------------------------------------------------------------------- #
# Primitive predicates (the lego set ground-truth rules are built from)        #
# --------------------------------------------------------------------------- #
def pred_marker(marker: str) -> Predicate:
    """True iff an event of type ``marker`` appears anywhere."""
    return lambda events: any(e["type"] == marker for e in events)


def pred_typed_successor(pairs: List[List[str]], lag: int) -> Predicate:
    """True iff some ``a`` is followed by its paired ``b`` within ``lag`` steps.

    ``lag`` is an index-distance window: b at index j counts for a at index i
    when ``i < j <= i + lag``. This is strictly stronger than mere presence —
    presence of a and b is not enough; b must follow a closely enough.
    """
    pairset = [tuple(p) for p in pairs]

    def f(events: List[Event]) -> bool:
        for i, e in enumerate(events):
            for a, b in pairset:
                if e["type"] == a:
                    hi = min(len(events), i + 1 + lag)
                    for j in range(i + 1, hi):
                        if events[j]["type"] == b:
                            return True
        return False

    return f


def pred_count_at_least(type: str, n: int) -> Predicate:
    """True iff ``type`` occurs at least ``n`` times."""
    return lambda events: sum(1 for e in events if e["type"] == type) >= n


def pred_request_count_at_least(n: int, exclude=("open", "close")) -> Predicate:
    """True iff at least ``n`` events occur whose type is not in ``exclude``.

    A volume predicate over the whole session (the rate/throttle signal), as
    opposed to ``count_at_least`` which counts one specific type.
    """
    ex = set(exclude)
    return lambda events: sum(1 for e in events if e["type"] not in ex) >= n


def pred_attr_count_at_least(key: str, value: Any, n: int) -> Predicate:
    """True iff at least ``n`` events carry ``attrs[key] == value`` (e.g. status 429)."""
    return lambda events: sum(
        1 for e in events if (e.get("attrs") or {}).get(key) == value) >= n


def pred_unguarded_action(actions: List[str], guard_type: str,
                          guard_key: Any = None, guard_value: Any = None) -> Predicate:
    """True iff some ``action`` event occurs with no qualifying *guard* before it.

    A precedence-and-absence primitive: an action of a type in ``actions`` is
    "unguarded" unless an earlier event of type ``guard_type`` (optionally with
    ``attrs[guard_key] == guard_value``) has already been seen. Captures
    precondition rules — e.g. a privileged write without a prior successful login.
    """
    actset = set(actions)

    def f(events: List[Event]) -> bool:
        guarded = False
        for e in events:
            if e["type"] == guard_type and (
                    guard_key is None or (e.get("attrs") or {}).get(guard_key) == guard_value):
                guarded = True
            elif e["type"] in actset and not guarded:
                return True
        return False

    return f


# --------------------------------------------------------------------------- #
# Rule families (labelers) keyed by planted_rule_id                            #
# --------------------------------------------------------------------------- #
def _marker_presence(params: Dict[str, Any]) -> Labeler:
    pred = pred_marker(params["marker"])
    hit, miss = params.get("hit", "reject"), params.get("miss", "approve")
    return lambda events: hit if pred(events) else miss


def _typed_successor(params: Dict[str, Any]) -> Labeler:
    pred = pred_typed_successor(params["pairs"], params["lag"])
    hit, miss = params.get("hit", "reject"), params.get("miss", "approve")
    return lambda events: hit if pred(events) else miss


def _channel_split(params: Dict[str, Any]) -> Labeler:
    """Disjoint sub-populations governed by different rules (Level 3)."""
    by = params["by"]
    default = params.get("default", "approve")
    subs = {key: make_labeler(spec["rule_id"], spec["params"])
            for key, spec in params["branches"].items()}

    def f(events: List[Event]) -> str:
        key = open_attr(events, by)
        if key in subs:
            return subs[key](events)
        return default

    return f


def _hierarchical(params: Dict[str, Any]) -> Labeler:
    """Outcome depends on the truth of sub-rules (Level 4).

    ``predicates`` maps a name to a predicate spec; ``table`` is an ordered list
    of rows ``{"cond": {name: bool, ...}, "outcome": str}`` — first row whose
    conditions all match wins; otherwise ``default``.
    """
    preds = {name: _build_predicate(spec) for name, spec in params["predicates"].items()}
    table = params["table"]
    default = params.get("default", "approve")

    def f(events: List[Event]) -> str:
        vals = {name: bool(p(events)) for name, p in preds.items()}
        for row in table:
            if all(vals[k] == v for k, v in row["cond"].items()):
                return row["outcome"]
        return default

    return f


def _decision_list(params: Dict[str, Any]) -> Labeler:
    """An ordered list of (condition -> outcome) clauses with a default.

    This is the inducer's hypothesis class: a legible, composable recombination
    of primitive predicates. A clause's condition is a **conjunction** of one or
    more predicates (``"all"``: every predicate must fire); a single-predicate
    clause may also use the shorthand ``"pred"``. The first clause whose condition
    holds decides the outcome; if none hold, the default applies.

    Conjunctions + ordering give negation for free, so this subsumes marker rules,
    multi-pair successor rules, and **hierarchies** (e.g. P1&P2 -> x ; P1 -> y ;
    P3 -> z ; else d, where clause 2 means "P1 and not P2"). It stays auditable.
    """
    compiled = []
    for c in params["clauses"]:
        if "all" in c:
            preds = [_build_predicate(p) for p in c["all"]]
            cond: Callable[[List[Event]], bool] = (
                lambda events, preds=preds: all(p(events) for p in preds))
        elif "any" in c:
            preds = [_build_predicate(p) for p in c["any"]]
            cond = lambda events, preds=preds: any(p(events) for p in preds)
        else:
            p0 = _build_predicate(c["pred"])
            cond = lambda events, p0=p0: p0(events)
        compiled.append((cond, c["outcome"]))
    default = params.get("default", "approve")

    def f(events: List[Event]) -> str:
        for cond, outcome in compiled:
            if cond(events):
                return outcome
        return default

    return f


def _counter_mod(params: Dict[str, Any]) -> Labeler:
    """Outcome = class[count(trigger) mod k] (Level 5).

    Supports two vocabularies expressing the *same* abstract structure: when
    ``by`` is set, the trigger type is chosen per case from ``vocab_triggers``
    keyed by ``open.attrs[by]``. This is the analogy test (Mechanism 4).
    """
    k = params["k"]
    classes = params["classes"]
    by = params.get("by")
    vocab_triggers = params.get("vocab_triggers")
    fixed_trigger = params.get("trigger")

    def f(events: List[Event]) -> str:
        if by is not None:
            trig = vocab_triggers.get(open_attr(events, by))
        else:
            trig = fixed_trigger
        count = sum(1 for e in events if e["type"] == trig)
        return classes[count % k]

    return f


_DISPATCH: Dict[str, Callable[[Dict[str, Any]], Labeler]] = {
    "R0_marker_presence": _marker_presence,
    "R1_typed_successor": _typed_successor,
    "R2_typed_successor_noisy": _typed_successor,   # same rule, dirtier data
    "R3_channel_split": _channel_split,
    "R4_hierarchical": _hierarchical,
    "R5_counter_mod_k": _counter_mod,
    "DL_decision_list": _decision_list,   # the inducer's hypothesis class
}

_PRED_DISPATCH: Dict[str, Callable[[Dict[str, Any]], Predicate]] = {
    "marker": lambda p: pred_marker(p["marker"]),
    "typed_successor": lambda p: pred_typed_successor(p["pairs"], p["lag"]),
    "count_at_least": lambda p: pred_count_at_least(p["type"], p["n"]),
    "attr_eq": lambda p: (lambda events: open_attr(events, p["key"]) == p["value"]),
    "request_count_at_least": lambda p: pred_request_count_at_least(
        p["n"], tuple(p.get("exclude", ("open", "close")))),
    "attr_count_at_least": lambda p: pred_attr_count_at_least(p["key"], p["value"], p["n"]),
    "unguarded_action": lambda p: pred_unguarded_action(
        p["actions"], p["guard_type"], p.get("guard_key"), p.get("guard_value")),
}

NO_RULE = "NEG_no_rule"   # the negative control has no labeler by design


def _build_predicate(spec: Dict[str, Any]) -> Predicate:
    return _PRED_DISPATCH[spec["pred"]](spec["params"])


def make_labeler(rule_id: str, params: Dict[str, Any]) -> Labeler:
    """Reconstitute an executable labeler from JSON ground truth."""
    if rule_id == NO_RULE:
        raise ValueError(
            "NEG_no_rule has no labeler: the negative control's outcomes are "
            "i.i.d. noise, independent of the events."
        )
    if rule_id not in _DISPATCH:
        raise KeyError(f"unknown planted_rule_id: {rule_id!r}")
    return _DISPATCH[rule_id](params)
